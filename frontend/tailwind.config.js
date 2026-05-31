/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          900: "#0b0f1a",
          800: "#111827",
          700: "#1f2937",
          600: "#374151",
        },
        brand: {
          500: "#6366f1",
          400: "#818cf8",
          300: "#a5b4fc",
        },
        accent: {
          500: "#22d3ee",
          400: "#67e8f9",
        },
      },
      boxShadow: {
        soft: "0 12px 30px -16px rgba(15, 23, 42, 0.9)",
        glow: "0 0 0 1px rgba(99, 102, 241, 0.2), 0 8px 24px rgba(99, 102, 241, 0.15)",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};