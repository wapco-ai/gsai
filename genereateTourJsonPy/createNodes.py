import os  
from PIL import Image  
from PIL.ExifTags import TAGS, GPSTAGS  

def get_geotagged_data(image_path):  
    try:  
        image = Image.open(image_path)  
        exif_data = image._getexif()  
        
        if not exif_data:  
            print(f"No EXIF data found for {image_path}.")  
            return None  
        
        # Convert the EXIF data to a usable format  
        decoded_exif = {}  
        for tag, value in exif_data.items():  
            decoded_tag = TAGS.get(tag, tag)  
            decoded_exif[decoded_tag] = value  

        # Extract GPS info if it exists  
        gps_info = decoded_exif.get('GPSInfo', None)  
        if not gps_info:  
            print(f"No GPSInfo found in EXIF data for {image_path}.")  
            return None  
        
        gps_data = {}  
        for key, value in gps_info.items():  
            if key in GPSTAGS:  
                gps_data[GPSTAGS[key]] = value  
        
        # Get latitude and longitude from GPS data  
        lat = gps_data.get('Latitude')  
        lon = gps_data.get('Longitude')  
        lat_ref = gps_data.get('LatitudeRef')  
        lon_ref = gps_data.get('LongitudeRef')  

        # Convert latitude and longitude to decimal degrees  
        if lat and lat_ref and lon and lon_ref:  
            latitude = convert_to_decimal(lat, lat_ref)  
            longitude = convert_to_decimal(lon, lon_ref)  
            return (latitude, longitude)  

        print(f"No complete GPS data found for {image_path}.")  
        return None  
    except Exception as e:  
        print(f"Error reading {image_path}: {e}")  
        return None  

def convert_to_decimal(coord, ref):  
    """Convert GPS coordinates to decimal format."""  
    degrees = coord[0]  
    minutes = coord[1] / 60.0  
    seconds = coord[2] / 3600.0  
    decimal_degrees = degrees + minutes + seconds  
    if ref in ['S', 'W']:  
        decimal_degrees = -decimal_degrees  
    return round(decimal_degrees, 6)  

def create_geotagged_image_array(image_folder):  
    image_data = []  
    
    for filename in os.listdir(image_folder):  
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Add other image formats if needed  
            print(f"Processing image: {filename}")  # Debug statement  
            image_path = os.path.join(image_folder, filename)  
            gps_coordinates = get_geotagged_data(image_path)  
            
            if gps_coordinates:  
                lat, lon = gps_coordinates  
                image_data.append({  
                    'filename': filename,  
                    'lat': lat,  
                    'lon': lon,  
                })  
            else:  
                print(f"No GPS data found for {filename}")  # Debug statement  

    return image_data  

# Example usage  
image_folder = r'D:\AI\imageProccessing\DCIM\apptest\motor'  # Update with your image folder path  
geotagged_images = create_geotagged_image_array(image_folder)  

# Check if we found any geotagged images  
if not geotagged_images:  
    print("No geotagged images found.")  
else:  
    # Print the array of geotagged images  
    for img in geotagged_images:  
        print(f"Image: {img['filename']}, Latitude: {img['lat']}, Longitude: {img['lon']}")