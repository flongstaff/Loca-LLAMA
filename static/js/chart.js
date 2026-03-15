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

// ── Chart color palette ──────────────────────────────────────────────

const BAR_PALETTE = [
  "#6c8cff", "#4caf50", "#ff9800", "#e91e63", "#00bcd4", "#9c27b0",
];

function getBarColor(index, customColors) {
  if (customColors && customColors[index]) return customColors[index];
  return BAR_PALETTE[index % BAR_PALETTE.length];
}

// ── Bar Chart ────────────────────────────────────────────────────────

/**
 * Draw a vertical bar chart on a canvas.
 * @param {HTMLCanvasElement} canvas
 * @param {Array<{label: string, value?: number, color?: string, segments?: Array<{label: string, value: number, color?: string}>}>} data
 * @param {{title?: string, unit?: string, height?: number, barColors?: string[]}} options
 */
export function drawBarChart(canvas, data, options = {}) {
  if (!canvas || !data || data.length === 0) return;

  const height = options.height || 220;
  const { ctx, displayWidth, displayHeight } = setupCanvas(canvas, height);
  ctx.clearRect(0, 0, displayWidth, displayHeight);

  const colors = getThemeColors();
  const pad = { top: 30, right: 20, bottom: 44, left: 55 };
  const w = displayWidth - pad.left - pad.right;
  const h = displayHeight - pad.top - pad.bottom;

  // Title
  if (options.title) {
    ctx.fillStyle = colors.textMuted;
    ctx.font = "12px -apple-system, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(options.title, pad.left, 16);
  }

  // Determine max Y from data (supports stacked segments)
  let maxY = 0;
  for (const item of data) {
    if (item.segments) {
      const total = item.segments.reduce((sum, s) => sum + s.value, 0);
      if (total > maxY) maxY = total;
    } else {
      if (item.value > maxY) maxY = item.value;
    }
  }
  if (maxY === 0) maxY = 1;
  // Add 15% headroom
  maxY = maxY * 1.15;

  // Y-axis guide lines
  const unit = options.unit || "";
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
    ctx.fillText(`${(maxY * frac).toFixed(1)}${unit}`, pad.left - 6, y + 4);
  }
  ctx.fillText(`0${unit}`, pad.left - 6, pad.top + h + 4);

  // Bar dimensions
  const barCount = data.length;
  const gap = Math.max(6, Math.min(20, w / barCount * 0.25));
  const barWidth = (w - gap * (barCount + 1)) / barCount;
  const clampedBarWidth = Math.min(barWidth, 60);
  const totalBarsWidth = clampedBarWidth * barCount + gap * (barCount + 1);
  const offsetX = pad.left + (w - totalBarsWidth) / 2 + gap;

  // Draw bars
  for (let i = 0; i < barCount; i++) {
    const item = data[i];
    const x = offsetX + i * (clampedBarWidth + gap);

    if (item.segments) {
      // Stacked bar
      let yOffset = 0;
      for (let s = 0; s < item.segments.length; s++) {
        const seg = item.segments[s];
        const segHeight = (seg.value / maxY) * h;
        const y = pad.top + h - yOffset - segHeight;

        ctx.fillStyle = seg.color || getBarColor(s, options.barColors);
        ctx.beginPath();
        // Rounded top only on the topmost segment
        if (s === item.segments.length - 1) {
          roundedRect(ctx, x, y, clampedBarWidth, segHeight, 3, true, false);
        } else if (s === 0) {
          roundedRect(ctx, x, y, clampedBarWidth, segHeight, 3, false, true);
        } else {
          ctx.rect(x, y, clampedBarWidth, segHeight);
        }
        ctx.fill();
        yOffset += segHeight;
      }

      // Value label above stacked bar
      const total = item.segments.reduce((sum, s) => sum + s.value, 0);
      const topY = pad.top + h - yOffset;
      ctx.fillStyle = colors.textMuted;
      ctx.font = "10px -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(total.toFixed(1) + unit, x + clampedBarWidth / 2, topY - 4);
    } else {
      // Simple bar
      const barHeight = (item.value / maxY) * h;
      const y = pad.top + h - barHeight;
      ctx.fillStyle = item.color || getBarColor(i, options.barColors);
      ctx.beginPath();
      roundedRect(ctx, x, y, clampedBarWidth, barHeight, 3, true, false);
      ctx.fill();

      // Value label above bar
      ctx.fillStyle = colors.textMuted;
      ctx.font = "10px -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(item.value.toFixed(1) + unit, x + clampedBarWidth / 2, y - 4);
    }

    // X-axis label below bar
    ctx.fillStyle = colors.textMuted;
    ctx.font = "10px -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.save();
    const labelX = x + clampedBarWidth / 2;
    const labelY = pad.top + h + 14;
    // Truncate long labels
    const maxLabelWidth = clampedBarWidth + gap - 4;
    let label = item.label;
    ctx.font = "10px -apple-system, sans-serif";
    if (ctx.measureText(label).width > maxLabelWidth) {
      while (label.length > 3 && ctx.measureText(label + "...").width > maxLabelWidth) {
        label = label.slice(0, -1);
      }
      label += "...";
    }
    ctx.fillText(label, labelX, labelY);
    ctx.restore();
  }
}

// ── Line Chart ───────────────────────────────────────────────────────

/**
 * Draw a line chart with optional dot markers.
 * @param {HTMLCanvasElement} canvas
 * @param {Array<{label: string, value: number}>} data
 * @param {{title?: string, unit?: string, height?: number, color?: string, showDots?: boolean}} options
 */
export function drawLineChart(canvas, data, options = {}) {
  if (!canvas || !data || data.length < 2) return;

  const height = options.height || 180;
  const { ctx, displayWidth, displayHeight } = setupCanvas(canvas, height);
  ctx.clearRect(0, 0, displayWidth, displayHeight);

  const colors = getThemeColors();
  const lineColor = options.color || colors.accent;
  const showDots = options.showDots !== false;
  const pad = { top: 30, right: 20, bottom: 34, left: 55 };
  const w = displayWidth - pad.left - pad.right;
  const h = displayHeight - pad.top - pad.bottom;

  // Title
  if (options.title) {
    ctx.fillStyle = colors.textMuted;
    ctx.font = "12px -apple-system, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(options.title, pad.left, 16);
  }

  // Determine Y range
  let maxY = Math.max(...data.map((d) => d.value));
  let minY = Math.min(...data.map((d) => d.value));
  // Add headroom
  const range = maxY - minY || 1;
  maxY = maxY + range * 0.1;
  minY = Math.max(0, minY - range * 0.1);
  const yRange = maxY - minY;

  // Y-axis guide lines
  const unit = options.unit || "";
  ctx.strokeStyle = colors.border;
  ctx.lineWidth = 1;
  ctx.font = "11px -apple-system, sans-serif";
  ctx.fillStyle = colors.textMuted;
  ctx.textAlign = "right";

  for (const frac of [0, 0.5, 1.0]) {
    const val = minY + frac * yRange;
    const y = pad.top + h - frac * h;
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + w, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillText(`${val.toFixed(1)}${unit}`, pad.left - 6, y + 4);
  }

  // Draw line
  const stepX = w / (data.length - 1);
  ctx.beginPath();
  data.forEach((d, i) => {
    const x = pad.left + i * stepX;
    const y = pad.top + h - ((d.value - minY) / yRange) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  ctx.stroke();

  // Fill area under line
  const lastX = pad.left + (data.length - 1) * stepX;
  ctx.lineTo(lastX, pad.top + h);
  ctx.lineTo(pad.left, pad.top + h);
  ctx.closePath();
  ctx.save();
  ctx.globalAlpha = 0.1;
  ctx.fillStyle = lineColor;
  ctx.fill();
  ctx.restore();

  // Dots
  if (showDots) {
    data.forEach((d, i) => {
      const x = pad.left + i * stepX;
      const y = pad.top + h - ((d.value - minY) / yRange) * h;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = lineColor;
      ctx.fill();
    });
  }

  // X-axis labels
  ctx.fillStyle = colors.textMuted;
  ctx.textAlign = "center";
  ctx.font = "10px -apple-system, sans-serif";

  // Show first, last, and middle labels
  const drawLabel = (idx) => {
    const x = pad.left + idx * stepX;
    ctx.fillText(data[idx].label, x, pad.top + h + 16);
  };

  drawLabel(0);
  drawLabel(data.length - 1);
  if (data.length > 4) {
    drawLabel(Math.floor(data.length / 2));
  }
}

// ── Helper: rounded rectangle ────────────────────────────────────────

function roundedRect(ctx, x, y, w, h, r, roundTop, roundBottom) {
  if (h < 1) { ctx.rect(x, y, w, h); return; }
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.moveTo(x + (roundTop ? r : 0), y);
  ctx.lineTo(x + w - (roundTop ? r : 0), y);
  if (roundTop) ctx.arcTo(x + w, y, x + w, y + r, r);
  else ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + h - (roundBottom ? r : 0));
  if (roundBottom) ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  else ctx.lineTo(x + w, y + h);
  ctx.lineTo(x + (roundBottom ? r : 0), y + h);
  if (roundBottom) ctx.arcTo(x, y + h, x, y + h - r, r);
  else ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + (roundTop ? r : 0));
  if (roundTop) ctx.arcTo(x, y, x + r, y, r);
  else ctx.lineTo(x, y);
}
