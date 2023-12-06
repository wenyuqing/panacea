# Panacea: Panoramic and Controllable Video Generation for Autonomous Driving

> [Paper] [**Panacea: Panoramic and Controllable Video Generation for Autonomous Driving**](https://arxiv.org/abs/2311.16813),            
> [WebPage] https://panacea-ad.github.io/
> Yuqing Wen<sup>1*&dagger;</sup>, Yucheng Zhao<sup>2*</sup>,Yingfei Liu<sup>2*</sup>, 
            Fan Jia<sup>2</sup>, 
            Yanhui Wang<sup>1</sup>, Chong Luo<sup>1</sup>,
            Chi Zhang<sup>3</sup>, 
            Tiancai Wang<sup>2&Dagger;</sup>, 
            Xiaoyan Sun<sup>1&Dagger;</sup>,
            Xiangyu Zhang<sup>2</sup> <br>
            <sup>1</sup>University of Science and Technology of China, 
            <sup>2</sup>MEGVII Technology, 
            <sup>3</sup>Mach Drive <br>
<sup>*</sup>Equal Contribution, 
            <sup>&dagger;</sup>This work was done during the internship at MEGVII, 
            <sup>&Dagger;</sup>Corresponding Author.
<div class="root-content" style="padding-top: 10px; width: 65%;">
        <h1 class="section-name">Generating <font style="color: red;">Multi-View and Controllable</font> Videos for Autonoumous Driving</h1>
        <img src="assests/pipeline.png" style="margin:auto; right: 0; left: 0; width: 90%; display: inline;">
        <p class="section-content-text" style="padding-bottom: 20px;"><strong>Overview of Panacea. </strong>(a). The diffusion training process of Panacea, enabled by a diffusion encoder and decoder with the decomposed 4D attention module. (b). The decomposed 4D attention module comprises three components: intra-view attention for spatial processing within individual views, cross-view attention to engage with adjacent views, and cross-frame attention for temporal processing. (c). Controllable module for the integration of diverse signals. The image conditions are derived from a frozen VAE encoder and combined with diffused noises. The text prompts are processed through a frozen CLIP encoder, while BEV sequences are handled via ControlNet. (d). The details of BEV layout sequences, including projected bounding boxes, object depths, road maps and camera pose.</p>
        <img src="assests/pipeline_inference.png" style="margin:auto; right: 0; left: 0; width: 65%; display: inline;">
        <p class="section-content-text" style="padding-bottom: 20px;"><strong>The two-stage inference pipeline of Panacea.</strong> Its two-stage process begins by creating multi-view images with BEV layouts, followed by using these images, along with subsequent BEV layouts, to facilitate the generation of following frames.</p>
</div>


