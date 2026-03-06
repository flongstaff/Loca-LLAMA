export function setupCanvas(canvas, displayHeight = 200) {
  const container = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const displayWidth = container.clientWidth;

  canvas.width = displayWidth * dpr;
  canvas.height = displayHeight * dpr;
  canvas.style.width = `${displayWidth}px`;
  canvas.style.height = `${displayHeight}px`;

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  return { ctx, displayWidth, displayHeight };
}

export function getThemeColors() {
  const style = getComputedStyle(document.documentElement);
  return {
    border: style.getPropertyValue("--border").trim() || "#2d3148",
    textMuted: style.getPropertyValue("--text-muted").trim() || "#8b8fa8",
    accent: style.getPropertyValue("--accent").trim() || "#6c8cff",
  };
}

export function drawGuideLines(ctx, pad, w, h, maxY, unit = "G") {
  const colors = getThemeColors();
  ctx.strokeStyle = colors.border;
  ctx.lineWidth = 1;
  ctx.font = "11px -apple-system, sans-serif";
  ctx.fillStyle = colors.textMuted;
  ctx.textAlign = "right";

  for (const frac of [0.25, 0.5, 0.75, 1.0]) {
    const y = pad.top + h - frac * h;
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + w, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillText(`${(maxY * frac).toFixed(0)}${unit}`, pad.left - 6, y + 4);
  }

  ctx.fillText(`0${unit}`, pad.left - 6, pad.top + h + 4);
}

export function drawAreaLine(ctx, pad, w, h, maxY, values, maxSlots, color) {
  if (values.length < 2) return;

  const stepX = w / (maxSlots - 1);
  const offset = maxSlots - values.length;

  ctx.beginPath();
  values.forEach((val, i) => {
    const x = pad.left + (i + offset) * stepX;
    const y = pad.top + h - (val / maxY) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  ctx.stroke();

  const lastIdx = values.length - 1;
  const lastX = pad.left + (lastIdx + offset) * stepX;
  const firstX = pad.left + offset * stepX;
  ctx.lineTo(lastX, pad.top + h);
  ctx.lineTo(firstX, pad.top + h);
  ctx.closePath();
  ctx.save();
  ctx.globalAlpha = 0.15;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
}

export function drawTimeLabels(ctx, pad, w, h, timestamps, maxSlots, formatFn) {
  if (timestamps.length < 2) return;

  const colors = getThemeColors();
  const stepX = w / (maxSlots - 1);
  const offset = maxSlots - timestamps.length;
  const lastIdx = timestamps.length - 1;

  ctx.fillStyle = colors.textMuted;
  ctx.textAlign = "center";
  ctx.font = "10px -apple-system, sans-serif";

  const firstX = pad.left + offset * stepX;
  const lastX = pad.left + (lastIdx + offset) * stepX;

  ctx.fillText(formatFn(timestamps[0]), firstX, pad.top + h + 16);
  ctx.fillText(formatFn(timestamps[lastIdx]), lastX, pad.top + h + 16);

  if (timestamps.length > 4) {
    const midIdx = Math.floor(lastIdx / 2);
    const midX = pad.left + (midIdx + offset) * stepX;
    ctx.fillText(formatFn(timestamps[midIdx]), midX, pad.top + h + 16);
  }
}
