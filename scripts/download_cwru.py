import os
import urllib.request
import time
import ssl

ssl_context = ssl._create_unverified_context()
BASE_URL = "https://engineering.case.edu/sites/default/files/{id}.mat"

DATA_IDS = {
    "normal": {
        "97": "Normal_0", "98": "Normal_1", "99": "Normal_2", "100": "Normal_3"
    },
    "12k_drive_end": {
        "105": "IR007_0", "106": "IR007_1", "107": "IR007_2", "108": "IR007_3",
        "118": "B007_0", "119": "B007_1", "120": "B007_2", "121": "B007_3",
        "130": "OR007@6_0", "131": "OR007@6_1", "132": "OR007@6_2", "133": "OR007@6_3",
        "144": "OR007@3_0", "145": "OR007@3_1", "146": "OR007@3_2", "147": "OR007@3_3",
        "156": "OR007@12_0", "158": "OR007@12_1", "159": "OR007@12_2", "160": "OR007@12_3",
        "169": "IR014_0", "170": "IR014_1", "171": "IR014_2", "172": "IR014_3",
        "185": "B014_0", "186": "B014_1", "187": "B014_2", "188": "B014_3",
        "197": "OR014@6_0", "198": "OR014@6_1", "199": "OR014@6_2", "200": "OR014@6_3",
        "209": "IR021_0", "210": "IR021_1", "211": "IR021_2", "212": "IR021_3",
        "222": "B021_0", "223": "B021_1", "224": "B021_2", "225": "B021_3",
        "234": "OR021@6_0", "235": "OR021@6_1", "236": "OR021@6_2", "237": "OR021@6_3",
        "246": "OR021@3_0", "247": "OR021@3_1", "248": "OR021@3_2", "249": "OR021@3_3",
        "258": "OR021@12_0", "259": "OR021@12_1", "260": "OR021@12_2", "261": "OR021@12_3",
        "3001": "IR028_0", "3002": "IR028_1", "3003": "IR028_2", "3004": "IR028_3",
        "3005": "B028_0", "3006": "B028_1", "3007": "B028_2", "3008": "B028_3"
    },
    "48k_drive_end": {
        "109": "IR007_0", "110": "IR007_1", "111": "IR007_2", "112": "IR007_3",
        "122": "B007_0", "123": "B007_1", "124": "B007_2", "125": "B007_3",
        "135": "OR007@6_0", "136": "OR007@6_1", "137": "OR007@6_2", "138": "OR007@6_3",
        "148": "OR007@3_0", "149": "OR007@3_1", "150": "OR007@3_2", "151": "OR007@3_3",
        "161": "OR007@12_0", "162": "OR007@12_1", "163": "OR007@12_2", "164": "OR007@12_3",
        "174": "IR014_0", "175": "IR014_1", "176": "IR014_2", "177": "IR014_3",
        "189": "B014_0", "190": "B014_1", "191": "B014_2", "192": "B014_3",
        "201": "OR014@6_0", "202": "OR014@6_1", "203": "OR014@6_2", "204": "OR014@6_3",
        "213": "IR021_0", "214": "IR021_1", "215": "IR021_2", "217": "IR021_3",
        "226": "B021_0", "227": "B021_1", "228": "B021_2", "229": "B021_3",
        "238": "OR021@6_0", "239": "OR021@6_1", "240": "OR021@6_2", "241": "OR021@6_3",
        "250": "OR021@3_0", "251": "OR021@3_1", "252": "OR021@3_2", "253": "OR021@3_3",
        "262": "OR021@12_0", "263": "OR021@12_1", "264": "OR021@12_2", "265": "OR021@12_3"
    },
    "12k_fan_end": {
        "270": "IR021_0", "271": "IR021_1", "272": "IR021_2", "273": "IR021_3",
        "274": "IR014_0", "275": "IR014_1", "276": "IR014_2", "277": "IR014_3",
        "278": "IR007_0", "279": "IR007_1", "280": "IR007_2", "281": "IR007_3",
        "282": "B007_0", "283": "B007_1", "284": "B007_2", "285": "B007_3",
        "286": "B014_0", "287": "B014_1", "288": "B014_2", "289": "B014_3",
        "290": "B021_0", "291": "B021_1", "292": "B021_2", "293": "B021_3",
        "294": "OR007@6_0", "295": "OR007@6_1", "296": "OR007@6_2", "297": "OR007@6_3",
        "298": "OR007@3_0", "299": "OR007@3_1", "300": "OR007@3_2", "301": "OR007@3_3",
        "302": "OR007@12_0", "305": "OR007@12_1", "306": "OR007@12_2", "307": "OR007@12_3",
        "309": "OR014@3_1", "310": "OR014@3_0", "311": "OR014@3_2", "312": "OR014@3_3",
        "313": "OR014@6_0", "315": "OR021@6_0", "316": "OR021@3_1", "317": "OR021@3_2", "318": "OR021@12_3"
    }
}

RAW_DATA_PATH = "data/raw"

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"Skipping {dest_path}, already exists.")
        return True
    
    try:
        print(f"Downloading {url} to {dest_path}...")
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, context=ssl_context, timeout=60) as response, open(dest_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    print("Starting CWRU dataset download...")
    total_files = sum(len(ids) for ids in DATA_IDS.values())
    current_file = 0
    
    for category, ids in DATA_IDS.items():
        cat_path = os.path.join(RAW_DATA_PATH, category)
        os.makedirs(cat_path, exist_ok=True)
        
        for fid, label in ids.items():
            current_file += 1
            url = BASE_URL.format(id=fid)
            dest_name = f"{label}.mat"
            dest_path = os.path.join(cat_path, dest_name)
            print(f"[{current_file}/{total_files}] Processing {label}...")
            success = download_file(url, dest_path)
            if not success:
                print(f"Failed to download {label} (ID: {fid})")
            
            time.sleep(0.5) # Be gentle with the server

    print("Download process completed.")

if __name__ == "__main__":
    main()
