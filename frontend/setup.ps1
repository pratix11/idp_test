# PropIntel Copilot — one-time setup
# Run this from the frontend/ directory: .\setup.ps1

Write-Host "Installing npm dependencies..." -ForegroundColor Cyan
npm install

Write-Host "Adding shadcn/ui components..." -ForegroundColor Cyan
npx shadcn@latest add --yes badge button card collapsible input scroll-area separator switch textarea sonner

Write-Host ""
Write-Host "Done! Start the dev server with:" -ForegroundColor Green
Write-Host "  npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "Make sure the FastAPI backend is running first:" -ForegroundColor Yellow
Write-Host "  uv run uvicorn property_intel.api.app:app --reload --port 8000" -ForegroundColor White
