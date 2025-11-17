# Deployment Guide

## Local Development

### Quick Start
```bash
# Install dependencies
npm install

# Start development server (runs on http://localhost:8080)
npm run dev
```

### Production Build (Local)
```bash
# Build for production
npm run build

# Preview production build
npm run preview

# Or serve with a simple HTTP server
npx serve -s dist -p 8080
```

## Docker Deployment

### Build and Run
```bash
# Build the Docker image
docker build -t omics-ui .

# Run the container
docker run -d -p 8080:80 --name omics-ui omics-ui

# Or use docker-compose
docker-compose up -d
```

### Stop Container
```bash
docker stop omics-ui
docker rm omics-ui

# Or with docker-compose
docker-compose down
```

## Cloud Hosting Options

### Vercel (Recommended - Easiest)
1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` from project directory
3. Follow prompts to deploy
4. Or connect GitHub repo at https://vercel.com

### Netlify
1. Install Netlify CLI: `npm i -g netlify-cli`
2. Build: `npm run build`
3. Deploy: `netlify deploy --prod --dir=dist`
4. Or drag & drop `dist` folder at https://app.netlify.com

### GitHub Pages
1. Add to `package.json` scripts:
   ```json
   "predeploy": "npm run build",
   "deploy": "gh-pages -d dist"
   ```
2. Install: `npm install --save-dev gh-pages`
3. Deploy: `npm run deploy`
4. Enable GitHub Pages in repo settings

### AWS S3 + CloudFront
1. Build: `npm run build`
2. Upload `dist` contents to S3 bucket
3. Configure bucket for static website hosting
4. Optionally add CloudFront CDN

### Self-Hosted with Nginx
1. Build: `npm run build`
2. Copy `dist` contents to `/var/www/omics-ui`
3. Use provided `nginx.conf` or configure nginx
4. Restart nginx: `sudo systemctl restart nginx`

## Environment Variables

Create `.env` file for environment-specific config:
```
VITE_API_URL=http://localhost:8000
VITE_APP_NAME=Multi-Omics Hypothesis Engine
```

Access in code: `import.meta.env.VITE_API_URL`

