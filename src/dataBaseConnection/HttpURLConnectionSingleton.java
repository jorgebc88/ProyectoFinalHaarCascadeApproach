package dataBaseConnection;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

public class HttpURLConnectionSingleton {
	private static HttpURLConnectionSingleton instance = null;
	private HttpURLConnection conn;
	public HttpURLConnection getConn() {
		return conn;
	}

	public void setConn(HttpURLConnection conn) {
		this.conn = conn;
	}

	protected HttpURLConnectionSingleton() throws IOException {
		   	String urlStr = "http://localhost:8080/FinalProject/detectedObject/DetectedObject";
		   	URL url = new URL(urlStr);
			this.conn = (HttpURLConnection) url.openConnection();
	   }

	public static HttpURLConnectionSingleton getInstance() {
		if (instance == null) {
			try {
				instance = new HttpURLConnectionSingleton();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return instance;
	}

}
