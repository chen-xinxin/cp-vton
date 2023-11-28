sh step1-train-gmm.sh
sh step2-test-gmm.sh
sh step3-generate-tom-data.sh
cp -r result/gmm_final-dresscode.pth/train/* data/train/
cp -r result/gmm_final-dresscode.pth/test/* data/test/
sh step4-train-tom.sh 
sh step5-test-tom.sh 
rm -r data/
cd ..
zip -r cpvton-dresscode-result.zip cp-vton-dresscode