
# MCD-Net: Towards RGB-D Video Inpainting in Real-World Scenes


Towards the goal of RGB-D video inpainting in real-world scenes, we make two major contributions including a *new* model and a *new* dataset.


**1. MCD-Net: using color and depth to mutually and implicitly inpaint each other**
![model](./figs/model.png)<br>
We integrate our proposed Depth Completion Network (*i.e.*, the JSCA and SRTA), Depth-Activated Transformer and Fusion Network into one framework for joint color and depth inpainting in RGB-D videos to achieve *SOTA accuracy and runtime*.

**2. VID Dataset: real RGB-D videos with dense annotated masks**
![dataset](./figs/dataset.png)<br>
We propose the first RGB-D video inpainting dataset (VID) with *authentic* RGB-D data and *elaborately-made masks* to support RGB-D video inpainting. A part of videos and masks in our VID dataset are placed at <a href="https://pan.baidu.com/s/1q9ys6ITxQgtfgYltQbdyvA?pwd=lor3" title="baidu" target="_blank">Baidu</a> or <a href="https://drive.google.com/file/d/1AXFX1FrmYZwo7dQIJD_SktBVTxrSXL4F/view?usp=sharing" title="google" target="_blank">Google Drive</a>.
![masks](./figs/mask.png)<br>
We *manually refine* the object masks automatically generated by <a href="https://github.com/gaomingqi/Track-Anything" title="track" target="_blank">Track-Anything</a>. The row 1 and 3 display raw masks. Raw 2 and 4 display our manually-corrected masks to ensure visually-pleasing object removal. We hope our VID dataset with our prepared object masks and occlusion masks can provide a more comprehensive, accurate and close-to-practice evaluation for RGB-D video inpainting.

## :newspaper: Abstract
<p align="left">
Video inpainting gains an increasing amount of attention ascribed to its wide applications in intelligent video editing. However, despite tremendous progress made in RGB video inpainting, the existing RGB-D video inpainting models are still incompetent to inpaint real-world RGB-D videos, as they simply fuse color and depth via explicit feature concatenation, neglecting the natural modality gap between color and depth. Moreover, current RGB-D video inpainting datasets are synthesized with homogeneous and delusive RGB-D data, which is far from real-world application and cannot provide comprehensive evaluation. To alleviate these problems and achieve real-world RGB-D video inpainting, on one hand, we propose a Mutually-guided Color and Depth Inpainting Network (MCD-Net), where color and depth are reciprocally leveraged to inpaint each other implicitly, mitigating the modal gap and fully exploiting cross-modal association for inpainting. On the other hand, we build a Video Inpainting with Depth (VID) dataset to supply diverse and authentic RGB-D video data with various object annotation masks to enable comprehensive evaluation for RGB-D video inpainting under real-world scenes. Experimental results on the DynaFill benchmark and our collected VID dataset demonstrate our MCD-Net not only yields the state-of-the-art quantitative performance but successfully achieves high-quality RGB-D video inpainting under real-world scenes. 
</p>

## :star: A video example of in-the-wild RGB-D video inpainting
![teaser](./demo/demo.gif#pic_left)

We feed an in-the-wild video (captured in SUSTech) to our model, and the model can be well-generalized to that, making flawless inpainted results.

## :star2: More video examples
We deliver several real-world video examples inpainted by our MCD-Net <a href="https://github.com/JCATCV/MCD-Net/tree/master/demo" title="demo" target="_blank">here</a>. <br>

![teaser](./figs/bicycle1.png)

![teaser](./figs/kid3.png)

![teaser](./figs/man1.png)

![teaser](./figs/girl1.png)


## :clap: Acknowledgement
We acknowledge <a href="https://github.com/gaomingqi/Track-Anything" title="track" target="_blank">Track-Anything</a>, <a href="https://github.com/robot-learning-freiburg/DynaFill/tree/master" title="dynafill" target="_blank">DynaFill</a>, <a href="https://github.com/MCG-NKU/E2FGVI" title="e2fgvi" target="_blank">E2FGVI</a>, <a href="https://github.com/ruiliu-ai/FuseFormer" title="fuseformer" target="_blank">Fuseformer</a>, <a href="https://github.com/researchmm/STTN" title="sttn" target="_blank">STTN</a>, <a href="https://github.com/seoungwugoh/opn-demo" title="opn" target="_blank">OPN</a>, <a href="https://github.com/vt-vl-lab/FGVC" title="fgvc" target="_blank">FGVC</a>, <a href="https://github.com/shleecs/Copy-and-Paste-Networks-for-Deep-Video-Inpainting" title="cpn" target="_blank">CAP</a>, <a href="https://github.com/ruiliu-ai/DSTT" title="dstt" target="_blank">DSTT</a>, <a href="https://github.com/kocchop/depth-completion-gan" title="dstt" target="_blank">SPGAN</a> and <a href="https://github.com/youmi-zym/CompletionFormer" title="cf" target="_blank">CompletionFormer</a> for their awesome works and the spirit of open source !
