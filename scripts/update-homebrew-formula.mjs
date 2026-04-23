#!/usr/bin/env node

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const args = process.argv.slice(2);

function getArg(name, { required = false } = {}) {
  const flag = `--${name}`;
  const index = args.findIndex((arg) => arg === flag || arg.startsWith(`${flag}=`));
  if (index === -1) {
    if (required) {
      throw new Error(`Missing required argument: --${name}`);
    }
    return undefined;
  }

  const value = args[index].includes("=") ? args[index].split("=")[1] : args[index + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`Argument --${name} requires a value`);
  }
  return value;
}

const version = getArg("version", { required: true });
const armUrl = getArg("arm-url", { required: true });
const armSha = getArg("arm-sha", { required: true });
const output = getArg("output", { required: true });

const templatePath = path.resolve(__dirname, "../packaging/homebrew/agent-context.rb");
const template = await readFile(templatePath, "utf8");
const rendered = template
  .replaceAll("__VERSION__", version)
  .replaceAll("__DARWIN_ARM64_URL__", armUrl)
  .replaceAll("__DARWIN_ARM64_SHA256__", armSha);

const outputPath = path.resolve(process.cwd(), output);
await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, rendered);
