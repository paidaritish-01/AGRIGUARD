"""
EXIF metadata extraction and GPS coordinate parsing for drone and mobile imagery.

CRITICAL REQUIREMENT:
Carries image GPS metadata through the pipeline, but explicitly documents that
sensor GPS coordinates reflect drone/camera position, not the exact location
of an individual plant or leaf.
"""

from typing import Dict, Optional, Tuple, Any
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


GPS_LIMITATION_NOTE = (
    "Drone/camera GPS metadata reflects the aircraft position and altitude at exposure time. "
    "It does not represent the exact ground location of an individual plant due to camera tilt, "
    "altitude, lens field-of-view, and surface topography."
)


def _convert_to_degrees(value: Any) -> Optional[float]:
    """Helper function to convert GPS coordinates stored in EXIF to decimal degrees."""
    try:
        if isinstance(value, (tuple, list)) and len(value) == 3:
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)
        return float(value)
    except Exception:
        return None


def extract_exif_gps(image: Image.Image) -> Dict[str, Any]:
    """
    Extract GPS latitude, longitude, and altitude from image EXIF data if present.
    """
    gps_info: Dict[str, Any] = {
        "latitude": None,
        "longitude": None,
        "altitude_m": None,
        "has_gps": False,
        "note": GPS_LIMITATION_NOTE,
    }

    try:
        exif = image.getexif()
        if not exif:
            return gps_info

        # Search for GPSInfo tag (tag ID 34853 = 0x8825)
        gps_tag_id = None
        for tag_id in exif:
            if TAGS.get(tag_id) == "GPSInfo":
                gps_tag_id = tag_id
                break

        if gps_tag_id is None:
            return gps_info

        gps_data = exif.get_ifd(gps_tag_id)
        if not gps_data:
            return gps_info

        # Parse GPS data dictionary
        parsed_gps = {}
        for key, val in gps_data.items():
            sub_name = GPSTAGS.get(key, key)
            parsed_gps[sub_name] = val

        lat = _convert_to_degrees(parsed_gps.get("GPSLatitude"))
        lat_ref = parsed_gps.get("GPSLatitudeRef", "N")
        lon = _convert_to_degrees(parsed_gps.get("GPSLongitude"))
        lon_ref = parsed_gps.get("GPSLongitudeRef", "E")

        if lat is not None and lon is not None:
            if lat_ref == "S":
                lat = -lat
            if lon_ref == "W":
                lon = -lon
            gps_info["latitude"] = round(lat, 6)
            gps_info["longitude"] = round(lon, 6)
            gps_info["has_gps"] = True

        alt = parsed_gps.get("GPSAltitude")
        if alt is not None:
            try:
                gps_info["altitude_m"] = round(float(alt), 2)
            except Exception:
                pass

    except Exception:
        # Gracefully handle corrupt or non-standard EXIF structures
        pass

    return gps_info
