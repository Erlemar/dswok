const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Configuration
const baseUrl = 'https://dswok.com/';
const rootDir = __dirname;
const outputFile = path.join(rootDir, 'sitemap.xml');

// Directories and files to exclude
const excludeDirs = [
  '.git',
  '.idea',
  '.obsidian',
  '.smart-connections',
  '.claude',
  'smart-chats',
  'update_later',
  'images'
];

// Individual files to exclude (meta files, README, working docs)
const excludeFiles = [
  'README.md',
  'claude.md',
  'ideas_1.md',
  'improvement_plan.md',
  'missing_notes.md'
];

// Function to find all markdown files
function findMarkdownFiles(dir, fileList = []) {
  const files = fs.readdirSync(dir);

  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      // Skip excluded directories
      if (!excludeDirs.includes(file)) {
        findMarkdownFiles(filePath, fileList);
      }
    } else if (file.endsWith('.md') && !excludeFiles.includes(file)) {
      // Add markdown files to the list (unless in exclude list)
      fileList.push(filePath);
    }
  });

  return fileList;
}

// Load list of currently-staged files once, so files about to be committed
// get today's date instead of the stale "previous commit" date from git log.
let stagedFiles = new Set();
try {
  const staged = execSync('git diff --cached --name-only', {
    cwd: rootDir,
    encoding: 'utf8'
  }).trim();
  if (staged) {
    stagedFiles = new Set(staged.split('\n').filter(Boolean));
  }
} catch (e) {
  // Not in a git repo or git unavailable — fine, proceed without staged info
}

// Function to get last-modified date (ISO 8601) for the sitemap
function getLastMod(filePath) {
  const relPath = path.relative(rootDir, filePath);

  // If file is staged in the current commit, use today's date — otherwise
  // git log returns the PREVIOUS commit date for this file, which would be
  // stale the moment the new commit lands.
  if (stagedFiles.has(relPath)) {
    return new Date().toISOString().split('T')[0];
  }

  // Otherwise use the last commit date from git log
  try {
    const isoDate = execSync(
      `git log -1 --format=%cI -- "${filePath}"`,
      { cwd: rootDir, encoding: 'utf8' }
    ).trim();
    if (isoDate) {
      return isoDate.split('T')[0];
    }
  } catch (e) {
    // Not in git or git unavailable — fall through
  }

  // Fallback: filesystem mtime
  const stat = fs.statSync(filePath);
  return stat.mtime.toISOString().split('T')[0];
}

// Function to convert file path to URL
function filePathToUrl(filePath) {
  // Remove the root directory from the path
  let relativePath = filePath.substring(rootDir.length);
  
  // Replace backslashes with forward slashes (for Windows)
  relativePath = relativePath.replace(/\\/g, '/');
  
  // Remove leading slash if present
  if (relativePath.startsWith('/')) {
    relativePath = relativePath.substring(1);
  }
  
  // Remove .md extension
  relativePath = relativePath.replace(/\.md$/, '');
  
  // Encode URL components
  const urlPath = relativePath.split('/').map(component => 
    encodeURIComponent(component)
  ).join('/');
  
  return baseUrl + urlPath;
}

// Generate sitemap XML
function generateSitemap(files) {
  let xml = '<?xml version="1.0" encoding="UTF-8"?>\n';
  xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n';
  
  // Add home page
  xml += '  <url>\n';
  xml += `    <loc>${baseUrl}</loc>\n`;
  xml += '    <priority>1.0</priority>\n';
  xml += '  </url>\n';
  
  // Add all markdown files
  files.forEach(file => {
    const url = filePathToUrl(file);
    const lastmod = getLastMod(file);
    xml += '  <url>\n';
    xml += `    <loc>${url}</loc>\n`;
    xml += `    <lastmod>${lastmod}</lastmod>\n`;
    xml += '    <priority>0.8</priority>\n';
    xml += '  </url>\n';
  });
  
  xml += '</urlset>';
  return xml;
}

// Main function
function main() {
  try {
    console.log('Finding markdown files...');
    const markdownFiles = findMarkdownFiles(rootDir);
    console.log(`Found ${markdownFiles.length} markdown files.`);
    
    console.log('Generating sitemap...');
    const sitemap = generateSitemap(markdownFiles);
    
    console.log(`Writing sitemap to ${outputFile}...`);
    fs.writeFileSync(outputFile, sitemap);
    
    console.log('Sitemap generated successfully!');
  } catch (error) {
    console.error('Error generating sitemap:', error);
  }
}

// Run the script
main();