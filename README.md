# Panacea: Panoramic and Controllable Video Generation for Autonomous Driving
**Official Repository of Panacea.**

> [Paper] [**Panacea: Panoramic and Controllable Video Generation for Autonomous Driving**](https://arxiv.org/abs/2311.16813),            
Yuqing Wen<sup>1*&dagger;</sup>, Yucheng Zhao<sup>2*</sup>,Yingfei Liu<sup>2*</sup>, 
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

> [WebPage] https://panacea-ad.github.io/

<div class="root-content" style="padding-top: 10px; width: 65%;">
        <h1 class="section-name">Generating <font style="color: red;">Multi-View and Controllable</font> Videos for Autonoumous Driving</h1>
        <img src="assests/pipeline.png" style="margin:auto; right: 0; left: 0; width: 90%; display: inline;">
        <p class="section-content-text" style="padding-bottom: 20px;"><strong>Overview of Panacea. </strong>(a). The diffusion training process of Panacea, enabled by a diffusion encoder and decoder with the decomposed 4D attention module. (b). The decomposed 4D attention module comprises three components: intra-view attention for spatial processing within individual views, cross-view attention to engage with adjacent views, and cross-frame attention for temporal processing. (c). Controllable module for the integration of diverse signals. The image conditions are derived from a frozen VAE encoder and combined with diffused noises. The text prompts are processed through a frozen CLIP encoder, while BEV sequences are handled via ControlNet. (d). The details of BEV layout sequences, including projected bounding boxes, object depths, road maps and camera pose.</p>
        <img src="assests/pipeline_inference.png" style="margin:auto; right: 0; left: 0; width: 65%; display: inline;">
        <p class="section-content-text" style="padding-bottom: 20px;"><strong>The two-stage inference pipeline of Panacea.</strong> Its two-stage process begins by creating multi-view images with BEV layouts, followed by using these images, along with subsequent BEV layouts, to facilitate the generation of following frames.</p>
</div>
<div class="root-content" style="padding-top: 10px; width: 65%; padding-bottom: 10px;">
        <h1 class="section-name">&#127916;&nbsp;&nbsp; BEV-guided Video Generation &nbsp;&nbsp;&#127916;</h1>
        <table style="width: 100%;">
            <tbody>
                <tr class="result-row">
                    <td>
                        <img src="assests/demo1.gif">
                    </td>
                </tr>
                <tr class="result-row">
                  <td>
                      <img src="assests/demo2.gif">
                  </td>
                </tr>
            </tbody>
        </table>
        <p class="section-content-text"><strong>Controllable multi-view video generation. Panacea is able to generate realistic, controllable videos with good temporal and view consistensy.</strong></p>
</div>
<div class="root-content" style="padding-top: 10px;width: 65%; padding-bottom: 10px;">
    <h1 class="section-name">&#127902;&nbsp;&nbsp; Attribute Controllable Video Generation &nbsp;&nbsp;&#127902;</h1>
    <table style="width: 100%;">
        <tbody>
          <tr class="result-row">
            <td>
                <img src="assests/attribute.png">
            </td>
        </tr>
        </tbody>
    </table>
    <p class="section-content-text"><strong>Video generation with variable attribute controls, such as weather, time, and scene, which allows Panacea to simulate a variety of rare driving scenarios, including extreme weather conditions such as rain and snow, thereby greatly enhancing the diversity of the data.</strong></p>
</div>
<div class="root-content" style="padding-top: 10px;width: 65%; padding-bottom: 10px;">
  <h1 class="section-name">&#128293;&nbsp;&nbsp; Benefiting Autonomous Driving  &nbsp;&nbsp;&#128293;</h1>
  <table style="padding-left: 120px;width: 90%;">
      <tbody>
        <tr class="result-row">
          <td>
              <img src="assests/gain.png">
          </td>
      </tr>
      </tbody>
  </table>
  <p class="section-content-text"><strong> (a). Panoramic video generation based on BEV (Bird’s-Eye-View) layout sequence facilitates the establishment of a synthetic video dataset, which enhances perceptual tasks. (b). Producing panoramic videos with conditional images and BEV layouts can effectively elevate image-only datasets to video datasets, thus enabling the advancement of video-based perception techniques.</strong></p>
</div>
<div style="background-color: white; margin-right: auto; margin-left: auto;">
    <div class="root-content" style="padding-top: 10px; width: 65%; padding-bottom: 10px;">
        <div>
            <h1 class="section-name" style="margin-top: 30px; text-align: left; font-size: 25px;">
                BibTex
            </h1>
            <a name="bib"></a>
            <pre style="margin-top: 5px;" class="bibtex">
                <code>
@artical{@misc{wen2023panacea,
    title={Panacea: Panoramic and Controllable Video Generation for Autonomous Driving}, 
    author={Yuqing Wen and Yucheng Zhao and Yingfei Liu and Fan Jia and Yanhui Wang and Chong Luo and Chi Zhang and Tiancai Wang and Xiaoyan Sun and Xiangyu Zhang},
    year={2023},
    eprint={2311.16813},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
}</code></pre>
        </div>
        <div style="margin-bottom: 0px;">
            <h1 class="section-name" style="margin-top: 0px; margin-bottom: 10px; text-align: left; font-size: 25px;">
                Contact
            </h1>
            <p class="section-content-text">
                Feel free to contact us at <strong>wenyuqing AT mail.ustc.edu.cn</strong> or <strong>wangtiancai AT megvii.com</strong>
        </div>
    </div>
</div>

