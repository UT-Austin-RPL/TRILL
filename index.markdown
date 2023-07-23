---
layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <title>Learning from Human Teleoperation: Skill-acquisition Framework for Humanoid Loco-manipulation</title>


<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->
<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>

<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-5RB3JP5LNX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-5RB3JP5LNX');
</script>

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Titillium Web","HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }
  
  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
    font-size:24px;
  }
  h3 {
    font-weight:300;
  }

	
IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table 
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


</head>

<body data-gr-c-s-loaded="true">


<style>
a {
  color: #bf5700;
  text-decoration: none;
  font-weight: 500;
}
</style>


<style>
highlight {
  color: #ff0000;
  text-decoration: none;
}
</style>

<div id="primarycontent">
<center><h1><strong>Learning from Human Teleoperation: Skill-acquisition Framework for Humanoid Loco-manipulation</strong></h1></center>
<center><h2>
    <a href="https://mingyoseo.com/">Mingyo Seo</a>&nbsp;&nbsp;&nbsp;
    <a href="https://www.linkedin.com/in/stevehan2001/">Steve Han</a>&nbsp;&nbsp;&nbsp;
    <a href="https://www.linkedin.com/in/kyutae-sim-888593166">Kyutae Sim</a>&nbsp;&nbsp;&nbsp;
    <a href="https://sites.utexas.edu/hcrl/people/">Seung Hyeon Bang</a>&nbsp;&nbsp;&nbsp;
    <a href="https://sites.utexas.edu/hcrl/people/">Carlos Gonzalez</a><br>
    <a href="https://www.ae.utexas.edu/people/faculty/faculty-directory/sentis">Luis Sentis</a>&nbsp;&nbsp;&nbsp; 
    <a href="https://cs.utexas.edu/~yukez">Yuke Zhu</a>
  </h2>
  <h2>
    <a href="https://www.utexas.edu/">The University of Texas at Austin</a>&nbsp;&nbsp;&nbsp;
  </h2>
  <h2><a href="http://arxiv.org/abs/2209.09233">Paper</a> | Code (coming soon)</h2>
  </center>

 <center><p><span style="font-size:20px;"></span></p></center>
<!-- <p> -->
<!--   </p><table border="0" cellspacing="10" cellpadding="0" align="center">  -->
<!--   <tbody> -->
<!--   <tr> -->
<!--   <\!-- For autoplay -\-> -->
<!-- <iframe width="560" height="315" -->
<!--   src="https://www.youtube.com/embed/GCfs3DJ4aO4?autoplay=1&mute=1&loop=1" -->
<!--   autoplay="true" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>   -->
<!--   <\!-- No autoplay -\-> -->
<!-- <\!-- <iframe width="560" height="315" -\-> -->
<!-- <\!--   src="https://www.youtube.com/embed/GCfs3DJ4aO4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>   -\-> -->

<!-- </tr> -->
<!-- </tbody> -->
<!-- </table> -->

<!--
<table border="0" cellspacing="10" cellpadding="0" align="center"> 
  <tbody>
    <tr> 
      <td align="center" valign="middle">
        <iframe width="800" height="450" src="https://www.youtube.com/embed/PdT8vBv9Asg?showinfo=0&playlist=PdT8vBv9Asg&autoplay=1&loop=1" autoplay="true" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </td>
    </tr>
  </tbody>
</table>
-->

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
    <tr>
      <td align="center" valign="middle">
        <video muted autoplay loop width="800">
          <source src="./src/video/header.mp4"  type="video/mp4">
        </video>
      </td>
    </tr>
  </tbody>
</table>

<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
Will be added later
</p></td></tr></table>
</p>
  </div>
</p>

<hr>

<h1 align="center">Method Overview</h1>

  <table border="0" cellspacing="10" cellpadding="0" align="center"> 
    <tbody>
      <tr> 
        <td align="center" valign="middle">
          <a href="./src/figure/approach.png"><img src="./src/figure/approach.png" style="width:100%;"> </a>
        </td>
      </tr>
    </tbody>
  </table>

  <table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  Overview of TRILL. We present a learning framework that allows a human demonstrator to easily provide supervision via teleoperation of task-space commands.
  The robot control interface then executes these target commands through joint-torque actuation. Policies trained on demonstration data, utilizing the underlying controller, can be directly deployed through this interface. This hierarchy enables the learned policies to perform effectively on floating-base robot systems, in both simulated and real-world environments.
</p></td></tr></table>

  
<br><br><hr> <h1 align="center">Hierarchical Perceptive Locomotion Model</h1> <!-- <h2
align="center"></h2> --> <table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle"><a href="./src/figure/pipeline.png"> <img
src="./src/figure/pipeline.png" style="width:100%;"> </a></td>
</tr> </tbody> </table>

<table align=center width=800px><tr><td> <p align="justify" width="20%">The high-level navigation policy generates the target velocity command at 10Hz from the onboard RGB-D camera observation and robot heading. The target velocity command, including linear and angular velocities, is used as input to the low-level gait controller along with the buffer of recent robot states. The low-level gait policy predicts the joint-space actions as the desired joint positions at 38Hz and sends them to the quadruped robot for actuation. More implementation details can be found in <a href="https://github.com/UT-Austin-RPL/PRELUDE/blob/main/implementation.md">this page</a>.
</p></td></tr></table>
<br>

<hr>

<h1 align="center">Teleopreation of the humanoid robot</h1>

  <table align=center width=800px><tr><td> <p align="justify" width="20%">Will be added later
  </p></td></tr></table>

  <!--
  <table border="0" cellspacing="10" cellpadding="0" align="center"> 
    <tbody>
      <tr> 
        <td align="center" valign="middle">
          <iframe width="798" height="300" src="https://www.youtube.com/embed/csr5hi5v_Bs?autoplay=1&mute=1&playlist=csr5hi5v_Bs&loop=1" autoplay="true" frameborder="5" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </td>
      </tr>
    </tbody>
  </table>
  -->
  
  <table border="0" cellspacing="10" cellpadding="0" align="center">
    <tbody>
      <tr>
        <td align="center" valign="middle">
          <video muted controls width="798">
            <source src="./src/video/evaluation.mp4"  type="video/mp4">
          </video>
        </td>
      </tr>
    </tbody>
  </table>

<hr>

<h1 align="center">Real Robot Evaluation</h1>

  <table align=center width=800px><tr><td> <p align="justify" width="20%">Will be added later
  </p></td></tr></table>

  <!--
  <table border="0" cellspacing="10" cellpadding="0" align="center"> 
    <tbody>
      <tr> 
        <td align="center" valign="middle">
          <iframe width="798" height="300" src="https://www.youtube.com/embed/csr5hi5v_Bs?autoplay=1&mute=1&playlist=csr5hi5v_Bs&loop=1" autoplay="true" frameborder="5" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </td>
      </tr>
    </tbody>
  </table>
  -->
  
  <table border="0" cellspacing="10" cellpadding="0" align="center">
    <tbody>
      <tr>
        <td align="center" valign="middle">
          <video muted controls width="798">
            <source src="./src/video/evaluation.mp4"  type="video/mp4">
          </video>
        </td>
      </tr>
    </tbody>
  </table>


<hr>

<h1 align="center">Simulation Evaluation</h1>

  <table align=center width=800px><tr><td> <p align="justify" width="20%">
  Will be added later
  </p></td></tr></table>


  <!--
  <table border="0" cellspacing="10" cellpadding="0" align="center"> 
    <tr>
        <td align="center" valign="middle">
          <iframe width="600" height="337" src="https://www.youtube.com/embed/K3pYobHhzDs?autoplay=1&mute=1&playlist=K3pYobHhzDs&loop=1" autoplay="true" frameborder="5" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </td>
      </tr>
      <tr>
        <td align="center" valign="middle">
          <iframe width="600" height="337" src="https://www.youtube.com/embed/cc3b8VM7Jb0?autoplay=1&mute=1&playlist=cc3b8VM7Jb0&loop=1" autoplay="true" frameborder="5" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </td>
      </tr>
      <tr>
        <td align="center" valign="middle">
          <iframe width="600" height="337" src="https://www.youtube.com/embed/9yEtgGHy9Aw?autoplay=1&mute=1&playlist=9yEtgGHy9Aw&loop=1" autoplay="true" frameborder="5" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </td>
      </tr>
  </table>
  -->
  <table border="0" cellspacing="10" cellpadding="0" align="center">
    <tbody>
      <tr>
        <td align="center" valign="middle">
          <video muted controls width="394">
            <source src="./src/video/deploy1.mp4"  type="video/mp4">
          </video>
        </td>

        <td align="center" valign="middle">
          <video muted controls width="394">
            <source src="./src/video/deploy2.mp4"  type="video/mp4">
          </video>
        </td>
      </tr>

      <tr>
        <td align="center" valign="middle">
          <video muted controls width="394">
            <source src="./src/video/deploy3.mp4"  type="video/mp4">
          </video>
        </td>

        <td align="center" valign="middle">
          <video muted controls width="394">
            <source src="./src/video/deploy4.mp4"  type="video/mp4">
          </video>
        </td>
      </tr>

    </tbody>
  </table>


<hr>
<center><h1>Citation</h1></center>

<table align=center width=800px>
  <tr>
    <td>
    <!-- <left> -->
    <pre><code style="display:block; overflow-x: auto">
      WILL BE ADDED LATER!!
    </code></pre>
    <!-- </left> -->
    </td>
  </tr>
</table>

