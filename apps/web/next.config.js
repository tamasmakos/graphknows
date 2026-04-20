/** @type {import('next').NextConfig} */
const nextConfig = {
    output: "standalone",
    experimental: {
        typedRoutes: false,
    },
};

module.exports = nextConfig;
