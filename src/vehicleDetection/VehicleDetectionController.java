package vehicleDetection;

import java.io.ByteArrayInputStream;
import java.util.Date;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import com.sun.org.apache.xml.internal.security.utils.JavaUtils;

import dataBaseConnection.ObjectRandom;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

/**
 * The controller associated with the only view of our application. The
 * application logic is implemented here. It handles the button for
 * starting/stopping the camera, the acquired video stream, the relative
 * controls and the face detection/tracking.
 * 
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @version 1.1 (2015-11-10)
 * @since 1.0 (2014-01-10)
 * 
 */
public class VehicleDetectionController {
	// FXML buttons
	@FXML
	private Button cameraButton;
	// the FXML area for showing the current frame
	@FXML
	private ImageView originalFrame;
	// checkboxes for enabling/disabling a classifier
	@FXML
	private CheckBox haarClassifier;
	@FXML
	private CheckBox lbpClassifier;

	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that performs the video capture
	private VideoCapture capture;
	// a flag to change the button behavior
	private boolean cameraActive;

	// face cascade classifier
	private CascadeClassifier faceCascade;
	private int absoluteFaceSize;

	/**
	 * Init the controller, at start time
	 */
	protected void init() {
		this.capture = new VideoCapture();
		this.faceCascade = new CascadeClassifier();
		this.absoluteFaceSize = 0;
	}

	/**
	 * The action triggered by pushing the button on the GUI
	 */
	@FXML
	protected void startCamera() {
		// set a fixed width for the frame
		originalFrame.setFitWidth(1080);
		// preserve image ratio
		originalFrame.setPreserveRatio(true);

		if (!this.cameraActive) {
			// disable setting checkboxes
			this.haarClassifier.setDisable(true);
			this.lbpClassifier.setDisable(true);

			// start the video capture
			this.capture.open("resources/video/MVI_3480.avi");

			// is the video stream available?
			if (this.capture.isOpened()) {
				this.cameraActive = true;

				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {

					@Override
					public void run() {
						Image imageToShow = grabFrame();
						originalFrame.setImage(imageToShow);
					}
				};

				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

				// update the button content
				this.cameraButton.setText("Stop Camera");
			} else {
				// log the error
				System.err.println("Failed to open the camera connection...");
			}
		} else {
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.cameraButton.setText("Start Camera");
			// enable classifiers checkboxes
			this.haarClassifier.setDisable(false);
			this.lbpClassifier.setDisable(false);

			// stop the timer
			try {
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			} catch (InterruptedException e) {
				// log the exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}

			// release the camera
			this.capture.release();
			// clean the frame
			this.originalFrame.setImage(null);
		}
	}

	/**
	 * Get a frame from the opened video stream (if any)
	 * 
	 * @return the {@link Image} to show
	 */
	private Image grabFrame() {
		// init everything
		Image imageToShow = null;
		Mat frame = new Mat();

		// check if the capture is open
		if (this.capture.isOpened()) {
			try {
				// read the current frame
				this.capture.read(frame);
				this.yLinePosition = frame.height() * 0.6;

				// if the frame is not empty, process it
				if (!frame.empty()) {
					// face detection
					this.detectAndDisplay(frame);

					// convert the Mat object (OpenCV) to Image (JavaFX)
					imageToShow = mat2Image(frame);
				}

			} catch (Exception e) {
				// log the (full) error
				System.err.println("ERROR: " + e);
			}
		}

		return imageToShow;
	}

	/**
	 * Method for face detection and tracking
	 * 
	 * @param frame
	 *            it looks for faces in this frame
	 */
	private void detectAndDisplay(Mat frame) {
		MatOfRect faces = new MatOfRect();
		Mat grayFrame = new Mat();

		// convert the frame in gray scale
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		// equalize the frame histogram to improve the result
		Imgproc.equalizeHist(grayFrame, grayFrame);

		// compute minimum face size (20% of the frame height, in our case)
		if (this.absoluteFaceSize == 0) {
			int height = grayFrame.rows();
			if (Math.round(height * 0.2f) > 0) {
				this.absoluteFaceSize = Math.round(height * 0.2f);
			}
		}
		Imgproc.line(frame, new Point(0, this.yLinePosition), new Point(0.8 * (frame.width()), this.yLinePosition),
				new Scalar(0, 0, 250));

		// detect faces
		this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
				new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());
		// each rectangle in faces is a face: draw them!
		Rect[] facesArray = faces.toArray();
		for (int i = 0; i < facesArray.length; i++) {
			Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0), 3);
			Imgproc.drawMarker(frame, this.calculateMassCenterRectangle(facesArray[i]), new Scalar(0, 250, 0));
			try {
				this.defineVehicle(facesArray[i], frame.width());
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	private int autos = 0, motos = 0;

	private Set<Vehicle> vehicleSet = new HashSet<>();

	private double yLinePosition = 0;

	private void defineVehicle(Rect rect, int width) throws Exception {
		Vehicle vehicleAux;
		boolean isANewVehicle = true;
		Date date = new Date();

		if (vehicleSet.isEmpty()) {
			if (this.calculateMassCenterRectangle(rect).y < this.yLinePosition) {
				date.setTime(System.currentTimeMillis());
				vehicleAux = new Vehicle(this.calculateMassCenterRectangle(rect), rect.area(), false,
						this.isGoingDown(rect, width), date);
				vehicleSet.add(vehicleAux);
				return;
			}
		}
		for (Vehicle vehicle : vehicleSet) {
			// if(areTheSameSize(vehicle.getVehicleSize(),
			// Imgproc.contourArea(contour))){
			if (this.isMoving(vehicle.getMassCenterLocation(), this.calculateMassCenterRectangle(rect))) {
				isANewVehicle = false;
				vehicle.setMassCenterLocation(this.calculateMassCenterRectangle(rect));
				vehicle.setVehicleSize(rect.area());
				date.setTime(System.currentTimeMillis());
				vehicle.setDetectionDate(date);
				if (!vehicle.isGoingUp()) {
					if (!vehicle.isCounted() && this.shouldBeCounted(rect)) {
						vehicle.setCounted(true);
						this.autos++; // modificar!!
						//ObjectRandom.httpPost("Car", vehicle.getDetectionDate());
						System.out.println("Autos: " + this.autos + " " + vehicle);
						this.vehicleSet.remove(vehicle);
					}
				}
				break;
			}
		}
		if (isANewVehicle) {
			if (this.calculateMassCenterRectangle(rect).y < this.yLinePosition) {
				date.setTime(System.currentTimeMillis());
				vehicleAux = new Vehicle(this.calculateMassCenterRectangle(rect), rect.area(), false,
						this.isGoingDown(rect, width), date);
				vehicleSet.add(vehicleAux);
			}
		}
	}

	private Point calculateMassCenterRectangle(Rect rect) {
		double x, y;
		x = rect.x + (rect.width / 2);
		y = rect.y + (rect.height / 2);
		return new Point(x, y);

	}
	
	private Point calculateMassCenter(Mat contour) {
		Moments moment = Imgproc.moments(contour);
		double cx = moment.m10 / moment.m00;
		double cy = moment.m01 / moment.m00;
		return new Point(cx, cy);
	}

	private boolean isGoingDown(Rect rect, int frameWidth) {
		if (this.calculateMassCenterRectangle(rect).x < frameWidth * 0.7) {
			return true;
		}
		return false;
	}

	private boolean shouldBeCounted(Rect rect) {
		double newYPosition = (this.calculateMassCenterRectangle(rect)).y;
		if (newYPosition > this.yLinePosition) {
			return true;
		}
		return false;
	}

	private boolean isMoving(Point oldCalculateMassCenter, Point newMassCenterLocation) {
		if (newMassCenterLocation.x >= oldCalculateMassCenter.x * 0.80
				&& newMassCenterLocation.x <= oldCalculateMassCenter.x * 1.20) {
			if (newMassCenterLocation.y >= oldCalculateMassCenter.y * 0.8
					&& newMassCenterLocation.y <= oldCalculateMassCenter.y * 1.2) {
				return true;
			}
		}
		return false;
	}

	private boolean areTheSameSize(double oldVehicle, double newVehicle) {
		if (newVehicle >= oldVehicle * 0.80 && newVehicle <= oldVehicle * 1.20) {
			return true;
		}
		return false;

	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/**
	 * The action triggered by selecting the Haar Classifier checkbox. It loads
	 * the trained set to be used for frontal face detection.
	 */
	@FXML
	protected void haarSelected(Event event) {
		// check whether the lpb checkbox is selected and deselect it
		if (this.lbpClassifier.isSelected())
			this.lbpClassifier.setSelected(false);

		this.checkboxSelection("resources/lbpcascades/1_carCascade.xml");
	}

	/**
	 * The action triggered by selecting the LBP Classifier checkbox. It loads
	 * the trained set to be used for frontal face detection.
	 */
	@FXML
	protected void lbpSelected(Event event) {
		// check whether the haar checkbox is selected and deselect it
		if (this.haarClassifier.isSelected())
			this.haarClassifier.setSelected(false);

		this.checkboxSelection("resources/lbpcascades/1_carCascade.xml");
	}

	/**
	 * Method for loading a classifier trained set from disk
	 * 
	 * @param classifierPath
	 *            the path on disk where a classifier trained set is located
	 */
	private void checkboxSelection(String classifierPath) {
		// load the classifier(s)
		this.faceCascade.load(classifierPath);

		// now the video capture can start
		this.cameraButton.setDisable(false);
	}

	/**
	 * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
	 * 
	 * @param frame
	 *            the {@link Mat} representing the current frame
	 * @return the {@link Image} to show
	 */
	private Image mat2Image(Mat frame) {
		// create a temporary buffer
		MatOfByte buffer = new MatOfByte();
		// encode the frame in the buffer, according to the PNG format
		Imgcodecs.imencode(".png", frame, buffer);
		// build and return an Image created from the image encoded in the
		// buffer
		return new Image(new ByteArrayInputStream(buffer.toArray()));
	}

}
