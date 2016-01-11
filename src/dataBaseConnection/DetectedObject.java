package dataBaseConnection;

import java.io.Serializable;
import java.util.Date;


/**
 * Created by Marco on 18/04/2015.
 */
public class DetectedObject implements Serializable{

    /**
	 * 
	 */
	private static final long serialVersionUID = 2607403988611110166L;

    private Long id;

    private String direction;

    private String ObjectType;

    private Date date;
    
    public DetectedObject(){
    }
    public DetectedObject(String direction, String objectType, Date date) {
    	super();
    	this.direction = direction;
        ObjectType = objectType;
        this.date = date;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public void setDirection(String direction) {
        this.direction = direction;
    }

    public void setObjectType(String objectType) {
        ObjectType = objectType;
    }

    public void setDate(Date date) {
        this.date = date;
    }

    public Long getId() {
        return id;
    }

    public String getDirection() {
        return direction;
    }

    public String getObjectType() {
        return ObjectType;
    }

    public Date getDate() {
        return date;
    }
}
