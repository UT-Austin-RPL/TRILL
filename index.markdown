---
layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <title>Deep Imitation Learning for Humanoid Loco-manipulation through Human Teleoperation</title>


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
<center><h1><strong>Deep Imitation Learning for Humanoid Loco-manipulation<br>through Human Teleoperation</strong></h1></center>
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
  <!--
  <h2><a href="">Paper</a> | Code (coming soon)</h2>
  -->
  </center>

 <center><p><span style="font-size:20px;"></span></p></center>

<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
We tackle the problem of developing humanoid loco-manipulation skills with deep imitation learning. The challenge of collecting human demonstrations for humanoids, in conjunction with the difficulty of  policy training under a high degree of freedom, presents substantial challenges.
We introduce <b>TRILL</b>, a data-efficient framework for learning humanoid loco-manipulation policies from human demonstrations. 
In this framework, we collect human demonstration data through an intuitive Virtual Reality (VR) interface.
We employ the whole-body control formulation to transform task-space commands from human operators into the robot's joint-torque actuation while stabilizing its dynamics.
By employing high-level action abstractions tailored for humanoid robots, our method can efficiently learn complex loco-manipulation skills.
We demonstrate the effectiveness of TRILL in simulation and on a real-world robot for performing various types of tasks. 
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

  <table align=center width=800px><tr><td><p align="justify" width="20%">
    TRILL addresses the challenge of learning humanoid loco-manipulation. 
    We introduce a learning framework that facilitates teleoperated demonstrations with task-space commands provided by a human demonstrator. 
    The trained policies leverage human complexity and adaptability in decision-making to generate these commands.
    The robot control interface then executes these target commands through joint-torque actuation, complying with robot dynamics.
    This synergistic combination of imitation learning and whole-body control enables successful method implementation in both simulated and real-world environments.
  </p></td></tr></table>

  
<br><br><hr> <h1 align="center">Hierarchical Loco-manipulation Pipeline</h1> <!-- <h2
align="center"></h2> --> <table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle"><a href="./src/figure/pipeline.png"> <img
src="./src/figure/pipeline.png" style="width:100%;"> </a></td>
</tr> </tbody> </table>

<table align=center width=800px><tr><td> <p align="justify" width="20%">
  The trained policies generate the target task-space command at 20Hz from the onboard stereo camera observation and the robot's proprioceptive feedback. The robot control interface realizes the task-space commands and computes the desired joint torques at 100Hz and sends them to the humanoid robot for actuation.
</p></td></tr></table>
<br>

<hr>

<h1 align="center">Real-robot Teleoperation</h1>

  <table align=center width=800px><tr><td> <p align="justify" width="20%">
    We design an intuitive VR teleoperation system, which reduces the cognitive and physical burdens for human operators to provide task demonstration. As a result, our teleoperation approach can produce high-quality demonstration data while maintaining safe robot operation.
  </p></td></tr></table>
  
  <table border="0" cellspacing="10" cellpadding="0" align="center"> 
    <tbody>
      <tr> 
        <td align="center" valign="middle">
          <video controls width="798">
            <source src="./src/video/demo_ramen.mp4"  type="video/mp4">
          </video>
        </td>
      </tr>
    </tbody>
  </table>
 
<hr>

<h1 align="center">Real-robot Deployment</h1>

  <table align=center width=800px><tr><td> <p align="justify" width="20%">
    We demonstrate the application of TRILL on the real robot, deploying visuomotor policies trained for dexterous manipulation tasks. During evaluation, the robot performed each task 10 times in a row without rebooting and succeeded in 8 out of 10 trials in the <i>Tool pick-and-place</i> task and 9 out of 10 trials in the <i>Removing the spray cap</i> task, respectively.
  </p></td></tr></table>

  <table border="0" cellspacing="10" cellpadding="0" align="center"> 
    <tbody>
      <tr> 
        <td align="center" valign="middle">
          <video muted controls width="394">
            <source src="./src/video/deploy_box.mp4"  type="video/mp4">
          </video>
        </td>
        <td align="center" valign="middle">
          <video muted controls width="394">
            <source src="./src/video/deploy_cap.mp4"  type="video/mp4">
          </video>
        </td>
      </tr>
    </tbody>
  </table>


<hr>

<h1 align="center">Simulation Evaluation</h1>

  <table align=center width=800px><tr><td> <p align="justify" width="20%">
    We design two realistic simulation environments and evaluate the robot’s ability to successfully perform subtasks involving free-space locomotion, manipulation, and loco-manipulation. TRILL, a framework tailored to train humanoid robots, achieves success rates of 96% for free-space locomotion tasks, 80% for manipulation tasks, and 92% for loco-manipulation tasks.
  </p></td></tr></table>

  <table border="0" cellspacing="10" cellpadding="0" align="center">
    <tbody>
      <tr>
        <td align="center" valign="middle">
          <video muted controls width="394">
            <source src="./src/video/deploy_door.mp4"  type="video/mp4">
          </video>
        </td>
        <td align="center" valign="middle">
          <video muted controls width="394">
            <source src="./src/video/deploy_workbench.mp4"  type="video/mp4">
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
    <pre><code style="display:block; overflow-x: auto">
      @misc{seo2023trill,
        title={Deep Imitation Learning for Humanoid Loco-manipulation 
	  through Human Teleoperation},
        author={Seo, Mingyo and Han, Steve and Sim, Kyutae and Bang, Seung Hyeon
	  and Gonzalez, Carlos and Sentis, Luis and Zhu, Yuke},
        year={2023}
      }
    </code></pre>
    </td>
  </tr>
</table>

