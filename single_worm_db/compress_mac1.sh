python3 ~/Documents/GitHub/Multiworm_Tracking/cmd_scripts/compressMultipleFiles.py \
/Volumes/behavgenom_archive\$/thecus /Volumes/behavgenom_archive\$/MaskedVideos \
--json_file ./JSON_singleworm/swimming.json \
--is_copy_video --is_single_worm --max_num_process 21 \
--videos_list ./files2analyze/videos_swimming_1.txt

python3 ../cmd_scripts/compressMultipleFiles.py \
 /Volumes/behavgenom_archive\$/thecus /Volumes/behavgenom_archive\$/MaskedVideos \
 --json_file ./JSON_singleworm/on_food.json \
 --is_copy_video --is_single_worm --max_num_process 21 \
 --videos_list ./files2analyze/videos_agar_1.txt



