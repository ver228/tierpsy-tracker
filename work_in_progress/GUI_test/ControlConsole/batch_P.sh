JSON_FILE=''
CMD_PY='track_ExpLocal_rig.py'
MAIN_DIR='/Volumes/behavgenom$/Pratheeban/MaskedVideos'

for CURR_DIR in `find "$MAIN_DIR" ! -path "$MAIN_DIR" -type d`
do
	python3 "$CMD_PY" "$CURR_DIR/" "$JSON_FILE"
done