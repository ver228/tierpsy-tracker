EXAMPLES_DIR="/Users/ajaver/Google Drive/MWTracker_Example"
SCRIPT_DIR=/Users/ajaver/Documents/GitHub/Multiworm_Tracking/cmd_scripts

TEST1_DIR="$EXAMPLES_DIR/test_1"
TEST2_DIR="$EXAMPLES_DIR/test_2"
TEST3_DIR="$EXAMPLES_DIR/test_3"


function test1 {
echo "%%%% TEST1 %%%%"
chflags -R nouchg "$TEST1_DIR/MaskedVideos"
rm -Rf "$TEST1_DIR/MaskedVideos"
python3  "$SCRIPT_DIR/compressMultipleFiles.py" "$TEST1_DIR/RawVideos" "$TEST1_DIR/MaskedVideos"
}

function test2 {
echo "%%%% TEST2 %%%%"
chflags -R nouchg "$TEST2_DIR/MaskedVideos"
rm -Rf "$TEST2_DIR/MaskedVideos"

python3  "$SCRIPT_DIR/compressMultipleFiles.py" "$TEST2_DIR/RawVideos" "$TEST2_DIR/MaskedVideos" \
--json_file "$TEST2_DIR/test2.json" --pattern_include "*.avi"
}

function test3 {
echo "%%%% TEST3 %%%%"
rm -Rf "$TEST3_DIR/Results"
python3  "$SCRIPT_DIR/trackMultipleFiles.py" "$TEST3_DIR/MaskedVideos" --json_file "$TEST3_DIR/test3.json"
}

function clean_all {
rm -Rf "$TEST1_DIR/MaskedVideos"
rm -Rf "$TEST2_DIR/MaskedVideos"
rm -Rf "$TEST3_DIR/Results"
}

#test1
#test1
test3