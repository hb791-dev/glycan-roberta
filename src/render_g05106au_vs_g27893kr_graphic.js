const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const SVG_DIR = path.join(PROJECT_ROOT, "public_reports", "example_glycan_pairs", "svgs");
const OUTPUT_DIR = path.join(PROJECT_ROOT, "public_reports", "example_glycan_pairs");
const OUTPUT_PATH = path.join(OUTPUT_DIR, "G05106AU_G27893KR.png");

const TOP = {
  svgPath: path.join(SVG_DIR, "G05106AU.svg"),
  prefix: "Fuca1-2(GalNAca1-3)Galb1-3(",
  highlight: "NeuAca2-6",
  suffix: ")GalNAca",
  cartoonCallout: { x: 255, y: 90, width: 205, height: 154, rx: 52, ry: 52 },
};

const BOTTOM = {
  svgPath: path.join(SVG_DIR, "G27893KR.svg"),
  prefix: "Fuca1-2(GalNAca1-3)Galb1-3(",
  highlight: "Galb1-4GlcNAcb1-6",
  suffix: ")GalNAca",
  cartoonCallout: { x: 130, y: 770, width: 306, height: 154, rx: 52, ry: 52 },
};

const LEFT_MARGIN = 56;
const TOP_MARGIN = 36;
const GLYCAN_LEFT = 72;
const GLYCAN_WIDTH = 690;
const GLYCAN_BOX_HEIGHT = 430;
const GLYCAN_GAP = 116;
const TEXT_GAP = 18;
const CANVAS_MIN_WIDTH = 1714;
const CANVAS_BACKGROUND = "#ffffff";
const CALLOUT = "#c74436";
const TEXT_FONT = "Arial Bold 56";
const TEXT_COLOR = "#000000";

function ensureExists(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing required file: ${filePath}`);
  }
}

async function renderTextSegment(text) {
  return sharp({
    text: {
      text: `<span font="${TEXT_FONT}" foreground="${TEXT_COLOR}">${escapeMarkup(text)}</span>`,
      rgba: true,
      dpi: 72,
    },
  })
    .png()
    .toBuffer({ resolveWithObject: true });
}

function escapeMarkup(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function makeOutlineSvg(width, height, rx, ry) {
  return Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">
      <rect x="3" y="3" width="${width - 6}" height="${height - 6}" rx="${rx}" ry="${ry}"
        fill="none" stroke="${CALLOUT}" stroke-width="6"/>
    </svg>`,
  );
}

async function renderGlycan(svgPath) {
  return sharp(svgPath)
    .resize({ width: GLYCAN_WIDTH, height: GLYCAN_BOX_HEIGHT, fit: "contain", background: CANVAS_BACKGROUND })
    .png()
    .toBuffer({ resolveWithObject: true });
}

async function buildLineBuffers(config) {
  const [prefix, highlight, suffix] = await Promise.all([
    renderTextSegment(config.prefix),
    renderTextSegment(config.highlight),
    renderTextSegment(config.suffix),
  ]);
  return { prefix, highlight, suffix };
}

function lineWidth(buffers) {
  return buffers.prefix.info.width + buffers.highlight.info.width + buffers.suffix.info.width;
}

async function main() {
  ensureExists(TOP.svgPath);
  ensureExists(BOTTOM.svgPath);
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  const [topGlycan, bottomGlycan, topLine, bottomLine] = await Promise.all([
    renderGlycan(TOP.svgPath),
    renderGlycan(BOTTOM.svgPath),
    buildLineBuffers(TOP),
    buildLineBuffers(BOTTOM),
  ]);

  const maxLineWidth = Math.max(lineWidth(topLine), lineWidth(bottomLine));
  const canvasWidth = Math.max(CANVAS_MIN_WIDTH, LEFT_MARGIN * 2 + maxLineWidth + 40);
  const topGlycanTop = TOP_MARGIN;
  const topTextTop = topGlycanTop + GLYCAN_BOX_HEIGHT + TEXT_GAP;
  const bottomGlycanTop = topTextTop + topLine.prefix.info.height + 110;
  const bottomTextTop = bottomGlycanTop + GLYCAN_BOX_HEIGHT + TEXT_GAP;
  const canvasHeight = bottomTextTop + bottomLine.prefix.info.height + 90;

  const composites = [
    {
      input: topGlycan.data,
      left: GLYCAN_LEFT,
      top: topGlycanTop,
    },
    {
      input: bottomGlycan.data,
      left: GLYCAN_LEFT,
      top: bottomGlycanTop,
    },
    {
      input: makeOutlineSvg(TOP.cartoonCallout.width, TOP.cartoonCallout.height, TOP.cartoonCallout.rx, TOP.cartoonCallout.ry),
      left: TOP.cartoonCallout.x,
      top: TOP.cartoonCallout.y,
    },
    {
      input: makeOutlineSvg(BOTTOM.cartoonCallout.width, BOTTOM.cartoonCallout.height, BOTTOM.cartoonCallout.rx, BOTTOM.cartoonCallout.ry),
      left: BOTTOM.cartoonCallout.x,
      top: BOTTOM.cartoonCallout.y,
    },
  ];

  const linePlacements = [
    { top: topTextTop, buffers: topLine },
    { top: bottomTextTop, buffers: bottomLine },
  ];

  for (const placement of linePlacements) {
    const prefixLeft = LEFT_MARGIN;
    const highlightLeft = prefixLeft + placement.buffers.prefix.info.width;
    const suffixLeft = highlightLeft + placement.buffers.highlight.info.width;
    const boxTop = placement.top + 6;
    const boxHeight = Math.max(placement.buffers.highlight.info.height - 12, 32);
    composites.push(
      {
        input: placement.buffers.prefix.data,
        left: prefixLeft,
        top: placement.top,
      },
      {
        input: makeOutlineSvg(placement.buffers.highlight.info.width + 18, boxHeight + 14, 4, 4),
        left: highlightLeft - 9,
        top: boxTop - 7,
      },
      {
        input: placement.buffers.highlight.data,
        left: highlightLeft,
        top: placement.top,
      },
      {
        input: placement.buffers.suffix.data,
        left: suffixLeft,
        top: placement.top,
      },
    );
  }

  await sharp({
    create: {
      width: canvasWidth,
      height: canvasHeight,
      channels: 4,
      background: CANVAS_BACKGROUND,
    },
  })
    .composite(composites)
    .png()
    .toFile(OUTPUT_PATH);

  console.log(OUTPUT_PATH);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
