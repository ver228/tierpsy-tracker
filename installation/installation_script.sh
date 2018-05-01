#!/bin/bash
set -e

OS=$(uname -s)

MW_MAIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
OPENWORM_DIR=$MW_MAIN_DIR/../open-worm-analysis-toolbox
TIERPSYFEATURES_DIR=$MW_MAIN_DIR/../tierpsy-features
OPENCV_DIR=$MW_MAIN_DIR/../opencv
OPENCV_VER="3.2.0"

EXAMPLES_LINK="https://imperiallondon-my.sharepoint.com/personal/ajaver_ic_ac_uk/_layouts/15/guestaccess.aspx?guestaccesstoken=ldZ18fLY%2bzlu7XuO9mbKVdyiKoH4naiesqiLXWU4vGQ%3d&docid=0cec4e52f4ccf4d5b8bb3a737020fc12f&rev=1"
EXAMPLES_DIR="$MW_MAIN_DIR/tests"

function setup_modules {
	git clone https://github.com/openworm/open-worm-analysis-toolbox $OPENWORM_DIR || :
	cd $OPENWORM_DIR
	git pull origin HEAD || :
	
	USER_CONFIG=$OPENWORM_DIR/open_worm_analysis_toolbox/user_config.py
	if [ ! -f $USER_CONFIG ]; then
	mv $OPENWORM_DIR/open_worm_analysis_toolbox/user_config_example.txt $USER_CONFIG
	fi
	
	git clone https://github.com/ver228/tierpsy-features $TIERPSYFEATURES_DIR || :
	cd $TIERPSYFEATURES_DIR
	git pull origin HEAD || :
	
	cd $MW_MAIN_DIR
	chmod -R ugo+rx $MW_MAIN_DIR/../open-worm-analysis-toolbox

	cd $OPENWORM_DIR
	python3 setup.py develop
	
	cd $TIERPSYFEATURES_DIR
	python3 setup.py develop
	
	cd $MW_MAIN_DIR
	python3 setup.py develop
	python3 setup.py clean --all
}

function download_examples {
	curl -L $EXAMPLES_LINK -o test_data.zip
	rm -rf $EXAMPLES_DIR/data
	unzip test_data.zip -d $EXAMPLES_DIR
	rm test_data.zip
}

function link_desktop {
	if [[ "${OS}" -eq "Darwin" ]];
	then
		echo "Creating Desktop Link..."
		DESKTOLINK="$HOME/Desktop/TierpsyTracker.command"
	
		EXIT_CMD="osascript -e 'tell application "Terminal" to close first window' & exit"
		echo "python3 $MW_MAIN_DIR/cmd_scripts/TierpsyTrackerConsole.py; $EXIT_CMD" > $DESKTOLINK
		chmod 744 $DESKTOLINK
	#I haven't implemented a similar short cut for linux
	fi
} 


function exec_all {
	setup_modules
	link_desktop
}


case $1 in
	"")
	exec_all
	;;
	"--setup_modules")
	setup_modules
	;;
	"--link_desktop")
	link_desktop
	;;
	"--tests"|"--download_examples")
	download_examples
	;;
	*)
	echo "Exiting... Unrecognized argument: $1"
	exit 1
	;;
esac
