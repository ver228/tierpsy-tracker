JSON_FILE='/Volumes/behavgenom$-1/Camille/Recordings/no_traj_filert.json'
CMD_PY='track_ExpLocal_rig.py'
MAIN_DIR='/Volumes/behavgenom$-1/Camille/Recordings/MaskedVideos'

for CURR_DIR in `find "$MAIN_DIR" ! -path "$MAIN_DIR" -type d`
do
	python3 "$CMD_PY" "$CURR_DIR/" "$JSON_FILE"
done