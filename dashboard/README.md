# 📊 MachineLens Interactive Dashboard

A highly responsive, premium glassmorphic web dashboard built to visualize, drag, drop, and customize pre-computed MachineLens diagnostics plots.

---

## 🚀 How to Run the Dashboard

Since the dashboard retrieves JSON files asynchronously via `fetch` requests, standard browsers block these requests if opened directly via double-clicking `index.html` due to **CORS (Cross-Origin Resource Sharing)** security constraints.

To bypass this and run it flawlessly, execute a simple, lightweight local HTTP web server.

### Method 1: Using Python (Instant, No Installations Required)
Open your terminal in the `MachineLens2` root folder and run:

```bash
python -m http.server 8000
```

Then, open your browser and navigate to:
👉 **[http://localhost:8000/dashboard/](http://localhost:8000/dashboard/)**

### Method 2: Using Node.js (Live Server / Vite / etc.)
If you have `npm` installed, you can use `npx` to serve the current directory:

```bash
npx -y serve
```

And open the URL displayed in the terminal!

---
