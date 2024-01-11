# Recognition of actions from camera

Based on pretrained models for Moments in Time Dataset

Action recognition using the pre-trained models trained on [Moments in Time](http://moments.csail.mit.edu/).

### Execute the model real-time recognition

Before running the script download the model :
[ResNet50 pretrained on ImageNet](http://data.csail.mit.edu/soundnet/actions3/split2/moments_v2_RGB_imagenet_resnet3d50_segment16.pth.tar)
```
    python test_video_cam.py
```
This will open your webcam and display the action label as text
Try to change action to see the result

If you encoutering error with the ffmpeg video player change the bin path in the utils script

### Download the Models

* Clone the code from Github:
```
    git clone https://github.com/metalbubble/moments_models.git
    cd moments_models
```

![result](http://relation.csail.mit.edu/data/bolei_juggling.gif)

RESULT ON sample_data/bolei_juggling
0.982 -> juggling
0.003 -> flipping
0.003 -> spinning

### Reference

Mathew Monfort, Alex Andonian, Bolei Zhou, Kandan Ramakrishnan, Sarah Adel Bargal, Tom Yan, Lisa Brown, Quanfu Fan, Dan Gutfruend, Carl Vondrick, Aude Oliva. Moments in Time Dataset: one million videos for event understanding. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019. [pdf](https://arxiv.org/pdf/1801.03150.pdf), [bib](http://moments.csail.mit.edu/data/moments.bib)

Mathew Monfort, Kandan Ramakrishnan, Alex Andonian, Barry A McNamara, Alex Lascelles, Bowen Pan, Quanfu Fan, Dan Gutfreund, Rogerio Feris, Aude Oliva. Multi-Moments in Time: Learning and Interpreting Models for Multi-Action Video Understanding. arxiv preprint arXiv:1911.00232, 2019. [pdf](https://arxiv.org/pdf/1911.00232), [bib](http://moments.csail.mit.edu/multi_data/multi_moments.bib)


### Acknowledgements

The project is supported by MIT-IBM Watson AI Lab, IBM Research, the SystemsThatLearn@CSAIL / Ignite Grant and the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/ Interior Business Center (DOI/IBC) contract number D17PC00341.
