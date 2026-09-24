// Convert a 3DGS .ply into a compact .spz using Spark's own encoder, so the
// output is guaranteed to load in the viewer.
//
//   node pipeline/to_spz.mjs in.ply out.spz [maxSh=1]
//
// maxSh: spherical-harmonics degree to keep (0 = flat color, smallest file;
// 1 = subtle view-dependent shine; 3 = full, ~4x bigger).
import { readFile, writeFile } from "node:fs/promises";
import { transcodeSpz, SplatFileType } from "@sparkjsdev/spark";

const [, , input, output, maxShArg] = process.argv;
if (!input || !output) {
  console.error("usage: node pipeline/to_spz.mjs in.ply out.spz [maxSh]");
  process.exit(1);
}
const maxSh = maxShArg === undefined ? 1 : Number(maxShArg);

const fileBytes = new Uint8Array(await readFile(input));
const { fileBytes: spz, clippedCount } = await transcodeSpz({
  inputs: [{ fileBytes, fileType: SplatFileType.PLY, pathOrUrl: input }],
  maxSh,
});
await writeFile(output, spz);
console.log(
  `${input} (${(fileBytes.length / 1e6).toFixed(1)} MB) -> ${output} (${(spz.length / 1e6).toFixed(1)} MB), sh=${maxSh}, clipped=${clippedCount}`,
);
