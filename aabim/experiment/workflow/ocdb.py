from ocdb.api.OCDBApi import new_api, OCDBApi

api = new_api(server_url="https://ocdb.eumetsat.int")

cal_char_types = ["STRAY", "RADCAL", "POLAR", "THERMAL", "ANGULAR"]
sensorID = ""

elem = "POLAR"

cal_char_candidates = OCDBApi.fidrad_list_files(api, elem)

for f in cal_char_candidates:
    OCDBApi.fidrad_download_file(api, f, "/D/Downloads/")
