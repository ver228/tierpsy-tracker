JSON_FILE='/Volumes/behavgenom$/Camille/Recordings/compress_param.json'
CMD_PY='compress_ExpLocal_rig.py'
MAIN_DIR='/Volumes/D/Camille'
OUT_DIR='/Volumes/behavgenom$/Camille/Recordings'

for CURR_DIR in `find "$MAIN_DIR" ! -path "$MAIN_DIR" -type d`
do
	python3 "$CMD_PY" "$CURR_DIR/" "$OUT_DIR" "$JSON_FILE"
done