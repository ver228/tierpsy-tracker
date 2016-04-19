python3 ../cmd_scripts/trackMultipleFiles.py \
/Volumes/behavgenom_archive\$/MaskedVideos \
--json_file ./JSON_singleworm/swimming.json \
--is_single_worm --max_num_process 21 --end_point INT_SKE_ORIENT \
--videos_list ./files2analyze/masks_swimming_2.txt

python3 ../cmd_scripts/trackMultipleFiles.py \
/Volumes/behavgenom_archive\$/MaskedVideos \
--json_file ./JSON_singleworm/on_food.json \
--is_single_worm --max_num_process 21 --end_point INT_SKE_ORIENT \
--videos_list ./files2analyze/masks_agar_2.txt


