## 

This exercise is clone from  the Yolov7 Github repository and trained with custom dataset from Roboflow

### To run the training
`python train.py --weights weights/yolov7_training.pt
--data aquarium_data/data.yaml
--batch-size 16
--epochs 30
--device 0
--workers 1
--hyp data/hyp.scratch.custom.yaml`

Public dataset can be found on Roboflow:  https://universe.roboflow.com/ananth-v-/aquarium-demo
### To detect the video
`python detect.py --weights weights/fish_best.pt --conf 0.5 --source fish_vid.mp4 --view-img --nosave --no-trace`

#### Tried using another fish video online to check 
`python detect.py --weights weights/fish_best.pt --conf=0.5 --source fish_sample_video.mp4 --view-img --no-trace --nosave`


### To track and count the fish
`python detect_count_and_track_fish.py --weights weights/fish_best.pt --conf 0.50 --source fish_vid.mp4 --view-img --nosave --no-trace
`
####
