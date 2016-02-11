MAX_NUM_PROCESS=6

for SET_N in 8 9 10 13

do
	python3 trackExpLocalChecked.py \
	/Volumes/behavgenom_archive\$/MaskedVideos/nas207-3/Data/from\ pc207-$SET_N/ \
	--is_single_worm --json /Users/ajaver/Desktop/JSON_singleworm/on_food.json \
	--end_point SKE_ORIENT --max_num_process $MAX_NUM_PROCESS
done

