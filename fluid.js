/* A small, bounded-resolution WebGL color field. No geometry, textures or 3D model. */
(() => {
  const vertex = `attribute vec2 position;void main(){gl_Position=vec4(position,0.,1.);}`;
  const fragment = `precision highp float;
uniform vec2 resolution;
uniform float time;
uniform float seed;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*f*(f*(f*6.-15.)+10.);return mix(mix(hash(i),hash(i+vec2(1.,0.)),f.x),mix(hash(i+vec2(0.,1.)),hash(i+vec2(1.,1.)),f.x),f.y);}
float field(vec2 p){return noise(p)*.78+noise(p*2.03+5.7)*.22;}
void main(){
 vec2 uv=gl_FragCoord.xy/resolution;
 float t=time*.17+seed;
 vec2 p=uv*vec2(2.3,2.1);
 // Change the field itself: each region deforms instead of moving as a flat layer.
 vec2 q=vec2(field(p+vec2(t*.23,-t*.15)),field(p+vec2(4.2-t*.19,1.7+t*.22)));
 vec2 r=vec2(field(p+1.7*q+vec2(1.7,t*.15)),field(p+1.7*q+vec2(8.3,-t*.18)));
 float f=field(p+2.4*r+vec2(t*.08,0.));
 float sweep=sin(uv.x*3.7+uv.y*2.1+q.x*4.2+t*.32)*.5+.5;
 float gold=smoothstep(.42,.79,f*.77+sweep*.23);
 vec3 crimson=vec3(.38,.012,.045);
 vec3 red=vec3(.78,.025,.045);
 vec3 vermilion=vec3(.96,.17,.035);
 vec3 amber=vec3(1.,.56,.055);
 vec3 light=vec3(1.,.84,.36);
 vec3 col=mix(crimson,red,smoothstep(.22,.63,r.y));
 col=mix(col,vermilion,smoothstep(.3,.66,f)*.7);
 col=mix(col,amber,gold);
 col=mix(col,light,smoothstep(.75,.98,gold)*.68);
 // Quiet left edge protects the editorial text without a gray veil.
 col*=.72+.28*smoothstep(0.,.7,uv.x);
 float grain=(hash(gl_FragCoord.xy)-.5)/255.;
 gl_FragColor=vec4(col+grain,1.);
}`;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  document.querySelectorAll('.fluid-art').forEach((art, index) => {
    const canvas = document.createElement('canvas');
    canvas.className = 'fluid-canvas';
    canvas.setAttribute('aria-hidden', 'true');
    const gl = canvas.getContext('webgl', { alpha:false, antialias:false, depth:false, stencil:false, powerPreference:'low-power' });
    if (!gl) return;
    let program;
    const shaders = [];
    try {
      for (const [type, source] of [[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]]) {
        const shader=gl.createShader(type);shaders.push(shader);gl.shaderSource(shader,source);gl.compileShader(shader);
        if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) throw new Error('shader unavailable');
      }
      program=gl.createProgram();shaders.forEach(shader=>gl.attachShader(program,shader));gl.linkProgram(program);
      if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error('program unavailable');
    } catch {
      shaders.forEach(shader=>gl.deleteShader(shader));if(program)gl.deleteProgram(program);return;
    }
    shaders.forEach(shader=>gl.deleteShader(shader));
    gl.useProgram(program);
    const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
    const position=gl.getAttribLocation(program,'position');gl.enableVertexAttribArray(position);gl.vertexAttribPointer(position,2,gl.FLOAT,false,0,0);
    const resolution=gl.getUniformLocation(program,'resolution'), clock=gl.getUniformLocation(program,'time');
    gl.uniform1f(gl.getUniformLocation(program,'seed'),index*5.8+2.1);
    let visible=false, frame=0, elapsed=0, previous=0, lost=false;
    function draw(){if(lost)return;gl.uniform2f(resolution,canvas.width,canvas.height);gl.uniform1f(clock,elapsed+12);gl.drawArrays(gl.TRIANGLES,0,3);}
    function resize(){
      const {width,height}=art.getBoundingClientRect();
      const scale=Math.min(1,Math.sqrt(240000/Math.max(1,width*height)));
      canvas.width=Math.max(1,Math.round(width*scale));canvas.height=Math.max(1,Math.round(height*scale));
      gl.viewport(0,0,canvas.width,canvas.height);draw();
    }
    function tick(now){
      if(previous)elapsed+=Math.min((now-previous)/1000,.1);
      previous=now;draw();frame=requestAnimationFrame(tick);
    }
    function sync(){
      cancelAnimationFrame(frame);previous=0;
      if(!lost && visible && !document.hidden && !reduced.matches && !document.documentElement.classList.contains('motion-paused'))frame=requestAnimationFrame(tick);
    }
    art.prepend(canvas);resize();art.classList.add('fluid-webgl');
    new ResizeObserver(resize).observe(art);
    new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;sync();}).observe(art);
    document.addEventListener('context-motion-change',sync);
    document.addEventListener('visibilitychange',sync);
    reduced.addEventListener('change',sync);
    canvas.addEventListener('webglcontextlost',()=>{lost=true;cancelAnimationFrame(frame);art.classList.remove('fluid-webgl');});
  });
})();
