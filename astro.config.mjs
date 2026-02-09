import { defineConfig } from 'astro/config';

// Replace 'your-repo-name' with your actual GitHub repo name
// If deploying to https://username.github.io (user/org site), remove the base option
export default defineConfig({
  site: 'https://your-username.github.io',
  base: '/your-repo-name',
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
    },
  },
});
