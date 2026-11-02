'use strict';
const path = require('node:path'), fs = require('node:fs'), cp = require('node:child_process');
const canvas = require(process.env.CEMIVO_CANVAS_MODULE || '@napi-rs/canvas');
function binary(name) {
  if (process.env[name.toUpperCase()]) return process.env[name.toUpperCase()];
  try { return cp.execFileSync('where.exe', [name], {encoding:'utf8', windowsHide:true}).trim().split(/\r?\n/)[0]; } catch {}
  const packages = path.join(process.env.LOCALAPPDATA || '', 'Microsoft/WinGet/Packages');
  if (fs.existsSync(packages)) for (const dir of fs.readdirSync(packages).filter(d=>d.startsWith('Gyan.FFmpeg_'))) {
    for (const build of fs.readdirSync(path.join(packages,dir))) {
      const file=path.join(packages,dir,build,'bin',name+'.exe'); if(fs.existsSync(file)) return file;
    }
  }
  return name;
}
function fonts() {
  const candidates=[['segoeui.ttf','Film Sans'],['segoeuib.ttf','Film Bold'],['consola.ttf','Film Mono']];
  for(const[file,family]of candidates){const p=path.join(process.env.WINDIR||'C:/Windows','Fonts',file);if(fs.existsSync(p))canvas.GlobalFonts.registerFromPath(p,family);}
}
module.exports={...canvas, binary, fonts};
