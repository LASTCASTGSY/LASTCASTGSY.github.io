import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://lastcastgsy.github.io',
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
    },
  },
});
