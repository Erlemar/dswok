const fs = require('fs');
const path = require('path');

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
  'smart-chats',
  'update_later',
  'images',
  'Use_cases'
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
    } else if (file.endsWith('.md')) {
      // Add markdown files to the list
      fileList.push(filePath);
    }
  });
  
  return fileList;
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
    xml += '  <url>\n';
    xml += `    <loc>${url}</loc>\n`;
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