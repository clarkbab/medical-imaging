function _post(m){
  var msg=Object.assign({isStreamlitMessage:true},m);
  try{window.parent.postMessage(msg,'*');}catch(e){}
  try{if(window.top!==window.parent)window.top.postMessage(msg,'*');}catch(e){}
}
function setVal(v){_post({type:'streamlit:setComponentValue',dataType:'json',value:v});}
function setH(h){_post({type:'streamlit:setFrameHeight',height:h});}
setH(0);

var LABELS=['Unlabelled','Not at all confident','Not very confident','Neither','Fairly confident','Very confident'];
var _conf=0,_clin=false,_ox=0,_oy=0,_rendered=false,_panelTop=80;
var _draggingOx=false,_draggingOy=false;
var _eventId=0;
// Track pending values so _updatePanel doesn't reset sliders while a send is in-flight.
var _pendingOx=null,_pendingOy=null;
var _oxMin=-500,_oxMax=500,_oyMin=-500,_oyMax=500;

// Confidence/clinical only — does NOT include offsets to avoid clobbering drag state.
function _sendConf(){setVal({confidence:_conf,clinical:_clin,event_id:++_eventId});}
// Offset changes — includes offsets so Python can update the contour position.
function _sendOffset(){_pendingOx=_ox;_pendingOy=_oy;setVal({confidence:_conf,clinical:_clin,offset_x:_ox,offset_y:_oy,event_id:++_eventId});}
// Undo — no offset payload; Python pops the undo stack and sends back the previous offset.
function _sendUndo(){setVal({action:'undo',confidence:_conf,clinical:_clin,event_id:++_eventId});}

function _buildPanel(){
  var doc=window.parent.document;
  if(!doc.getElementById('_ca_conf_style')){
    var s=doc.createElement('style');
    s.id='_ca_conf_style';
    s.textContent=
      '#_ca_conf_panel{position:fixed;right:16px;background:#fff;'+
      'border:1px solid #ccc;border-radius:8px;overflow:hidden;'+
      'box-shadow:2px 4px 16px rgba(0,0,0,0.18);z-index:99999;'+
      'width:240px;font-family:sans-serif;font-size:13px;color:#222;}'+
      '#_ca_drag_h{background:#f5f5f5;border-bottom:1px solid #ddd;padding:4px 10px;'+
      'cursor:grab;text-align:center;font-size:13px;color:#bbb;user-select:none;letter-spacing:3px;}'+
      '#_ca_drag_h:active{cursor:grabbing;}'+
      '#_ca_body{padding:10px 14px 12px;}'+
      '#_ca_conf_panel h4{font-size:13px;font-weight:600;margin-bottom:7px;color:#333;}'+
      '#_ca_conf_panel .rlbl{display:flex;align-items:center;gap:6px;margin:3px 0;cursor:pointer;user-select:none;}'+
      '#_ca_conf_panel .sep{border-top:1px solid #eee;margin:8px 0;}'+
      '#_ca_conf_panel .clin{font-weight:600;}'+
      '#_ca_conf_panel .orow{display:flex;align-items:center;gap:3px;margin:4px 0;}'+
      '#_ca_conf_panel .olbl{width:14px;font-size:11px;color:#666;flex-shrink:0;}'+
      '#_ca_conf_panel .orow input[type=range]{flex:1;min-width:0;margin:0;}'+
      '#_ca_conf_panel .oval{width:30px;text-align:right;font-size:11px;color:#444;flex-shrink:0;}'+
      '#_ca_conf_panel .obtn{padding:1px 5px;font-size:11px;cursor:pointer;'+
      'border:1px solid #bbb;border-radius:3px;background:#f7f7f7;flex-shrink:0;line-height:1.4;}'+
      '#_ca_conf_panel .obtn:hover{background:#e0e0e0;}'+
      '#_ca_action_row{display:flex;gap:6px;margin-top:8px;}'+
      '#_ca_action_row button{flex:1;padding:4px;font-size:12px;cursor:pointer;'+
      'border:1px solid #bbb;border-radius:4px;background:#f7f7f7;}'+
      '#_ca_action_row button:hover:not(:disabled){background:#e0e0e0;}'+
      '#_ca_action_row button:disabled{opacity:0.4;cursor:default;}';
    doc.head.appendChild(s);
  }
  var panel=doc.getElementById('_ca_conf_panel');
  if(panel)return panel;
  panel=doc.createElement('div');
  panel.id='_ca_conf_panel';
  panel.style.top=_panelTop+'px';
  var h='<div id="_ca_drag_h">&#8943;&#8943;&#8943;</div><div id="_ca_body">';
  h+='<h4>Offset</h4>';
  h+='<div class="orow"><span class="olbl">X</span>'+
     '<button class="obtn" id="_ca_ox_dec">&#9664;</button>'+
     '<input type="range" id="_ca_ox_sl" min="-500" max="500" value="0">'+
     '<button class="obtn" id="_ca_ox_inc">&#9654;</button>'+
     '<span class="oval" id="_ca_ox_v">0</span></div>';
  h+='<div class="orow"><span class="olbl">Y</span>'+
     '<button class="obtn" id="_ca_oy_dec">&#9664;</button>'+
     '<input type="range" id="_ca_oy_sl" min="-500" max="500" value="0">'+
     '<button class="obtn" id="_ca_oy_inc">&#9654;</button>'+
     '<span class="oval" id="_ca_oy_v">0</span></div>';
  h+='<div id="_ca_action_row"><button id="_ca_undo_btn">&#8630; Undo last</button><button id="_ca_reset_btn">&#128260; Reset offsets</button></div>';
  h+='<div class="sep"></div>';
  h+='<h4>Confidence</h4>';
  for(var i=0;i<LABELS.length;i++)
    h+='<label class="rlbl"><input type="radio" name="_ca_r" value="'+i+'"> '+LABELS[i]+'</label>';
  h+='<div class="sep"></div>';
  h+='<label class="clin"><input type="checkbox" id="_ca_clin_cb"> Clinically acceptable</label>';
  h+='</div>';
  panel.innerHTML=h;
  doc.body.appendChild(panel);

  // Drag handle — vertical repositioning only.
  var dh=panel.querySelector('#_ca_drag_h');
  var _dg=false,_dStartY=0,_dPanelTop=0;
  dh.addEventListener('mousedown',function(e){
    _dg=true;_dStartY=e.clientY;_dPanelTop=parseInt(panel.style.top)||_panelTop;e.preventDefault();
  });
  doc.addEventListener('mousemove',function(e){
    if(!_dg)return;
    var t=_dPanelTop+(e.clientY-_dStartY);
    var _hdr=doc.querySelector('header[data-testid="stHeader"]');
    var _minTop=_hdr?Math.ceil(_hdr.getBoundingClientRect().bottom):0;
    t=Math.max(_minTop,Math.min(doc.documentElement.clientHeight-100,t));
    panel.style.top=t+'px';_panelTop=t;
  });
  doc.addEventListener('mouseup',function(){_dg=false;});

  // Confidence radios — send conf/clinical only, offsets untouched.
  panel.querySelectorAll('input[type="radio"]').forEach(function(r){
    r.addEventListener('change',function(){_conf=parseInt(this.value,10);_sendConf();});
  });
  // Clinical checkbox — send conf/clinical only.
  panel.querySelector('#_ca_clin_cb').addEventListener('change',function(){_clin=this.checked;_sendConf();});

  // X offset — update label on input, send offset on release.
  var oxsl=panel.querySelector('#_ca_ox_sl'),oxv=panel.querySelector('#_ca_ox_v');
  oxsl.addEventListener('mousedown',function(){_draggingOx=true;});
  oxsl.addEventListener('mouseup',function(){_draggingOx=false;});
  oxsl.addEventListener('input',function(){_ox=parseInt(this.value,10);oxv.textContent=_ox;});
  oxsl.addEventListener('change',function(){_ox=parseInt(this.value,10);oxv.textContent=_ox;_sendOffset();});
  panel.querySelector('#_ca_ox_dec').addEventListener('click',function(){
    _ox=Math.max(_oxMin,_ox-1);oxsl.value=_ox;oxv.textContent=_ox;_sendOffset();
  });
  panel.querySelector('#_ca_ox_inc').addEventListener('click',function(){
    _ox=Math.min(_oxMax,_ox+1);oxsl.value=_ox;oxv.textContent=_ox;_sendOffset();
  });

  // Y offset.
  var oysl=panel.querySelector('#_ca_oy_sl'),oyv=panel.querySelector('#_ca_oy_v');
  oysl.addEventListener('mousedown',function(){_draggingOy=true;});
  oysl.addEventListener('mouseup',function(){_draggingOy=false;});
  oysl.addEventListener('input',function(){_oy=parseInt(this.value,10);oyv.textContent=_oy;});
  oysl.addEventListener('change',function(){_oy=parseInt(this.value,10);oyv.textContent=_oy;_sendOffset();});
  panel.querySelector('#_ca_oy_dec').addEventListener('click',function(){
    _oy=Math.max(_oyMin,_oy-1);oysl.value=_oy;oyv.textContent=_oy;_sendOffset();
  });
  panel.querySelector('#_ca_oy_inc').addEventListener('click',function(){
    _oy=Math.min(_oyMax,_oy+1);oysl.value=_oy;oyv.textContent=_oy;_sendOffset();
  });

  // Reset button — explicitly zero both offsets and send.
  panel.querySelector('#_ca_reset_btn').addEventListener('click',function(){
    _ox=0;_oy=0;oxsl.value=0;oysl.value=0;oxv.textContent=0;oyv.textContent=0;_sendOffset();
  });
  // Undo button — Python pops the stack and sends back the previous offset.
  panel.querySelector('#_ca_undo_btn').addEventListener('click',function(){_sendUndo();});

  return panel;
}

function _updatePanel(conf,clin,ox,oy,oxMin,oxMax,oyMin,oyMax,undoAvailable){
  var p=_buildPanel();
  p.style.top=_panelTop+'px';
  p.querySelectorAll('input[type="radio"]').forEach(function(r){r.checked=parseInt(r.value,10)===conf;});
  var cb=p.querySelector('#_ca_clin_cb');
  if(cb)cb.checked=clin;
  // Update slider ranges from computed contour bounds.
  _oxMin=oxMin;_oxMax=oxMax;_oyMin=oyMin;_oyMax=oyMax;
  var oxsl=p.querySelector('#_ca_ox_sl'),oysl=p.querySelector('#_ca_oy_sl');
  oxsl.min=oxMin;oxsl.max=oxMax;oysl.min=oyMin;oysl.max=oyMax;
  // Don't overwrite sliders with stale Python args while a user send is in-flight.
  var okOx=!_draggingOx&&(_pendingOx===null||ox===_pendingOx);
  var okOy=!_draggingOy&&(_pendingOy===null||oy===_pendingOy);
  if(okOx){oxsl.value=ox;p.querySelector('#_ca_ox_v').textContent=ox;_ox=ox;_pendingOx=null;}
  if(okOy){oysl.value=oy;p.querySelector('#_ca_oy_v').textContent=oy;_oy=oy;_pendingOy=null;}
  _conf=conf;_clin=clin;
  var undoBtn=p.querySelector('#_ca_undo_btn');if(undoBtn)undoBtn.disabled=!undoAvailable;
}

window.addEventListener('message',function(e){
  if(!e.data||e.data.type!=='streamlit:render')return;
  _rendered=true;
  var a=e.data.args;
  _updatePanel(a.confidence||0,!!a.clinical,a.offset_x||0,a.offset_y||0,
               a.ox_min!==undefined?a.ox_min:-500,a.ox_max!==undefined?a.ox_max:500,
               a.oy_min!==undefined?a.oy_min:-500,a.oy_max!==undefined?a.oy_max:500,
               !!a.undo_available);
});

var _rm={type:'streamlit:componentReady',apiVersion:1};
_post(_rm);
var _rt=setInterval(function(){if(_rendered){clearInterval(_rt);return;}_post(_rm);},100);
