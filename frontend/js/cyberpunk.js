/**
 * NOXIS — Cyberpunk Background FX
 * Canvas-based particle system + DOM element injection
 * Uses existing purple accent palette (#8B5CF6)
 */
(function () {
    'use strict';

    /* ── Inject DOM elements ─────────────────────────────────── */
    function injectCyberElements() {
        const body = document.body;

        // Central radial glow
        const glow = document.createElement('div');
        glow.className = 'cyber-glow';
        body.insertBefore(glow, body.firstChild);

        // Corner brackets
        const corners = document.createElement('div');
        corners.className = 'cyber-corners';
        body.insertBefore(corners, body.firstChild);

        // Floating hex particles container
        const particles = document.createElement('div');
        particles.className = 'cyber-particles';
        body.insertBefore(particles, body.firstChild);

        // Spawn 18 floating hex particles
        for (let i = 0; i < 18; i++) {
            const hex = document.createElement('div');
            hex.className = 'hex';
            hex.style.left = Math.random() * 100 + '%';
            hex.style.animationDuration = (12 + Math.random() * 18) + 's';
            hex.style.animationDelay = (Math.random() * 15) + 's';
            hex.style.width = (4 + Math.random() * 8) + 'px';
            hex.style.height = hex.style.width;
            hex.style.background = `rgba(139, 92, 246, ${0.06 + Math.random() * 0.15})`;
            particles.appendChild(hex);
        }

        // Neon streak lines container
        const lines = document.createElement('div');
        lines.className = 'cyber-lines';
        body.insertBefore(lines, body.firstChild);

        // Spawn 5 horizontal streaks
        for (let i = 0; i < 5; i++) {
            const streak = document.createElement('div');
            streak.className = 'streak';
            streak.style.top = (15 + Math.random() * 70) + '%';
            streak.style.width = (100 + Math.random() * 250) + 'px';
            streak.style.animationDuration = (4 + Math.random() * 6) + 's';
            streak.style.animationDelay = (Math.random() * 10) + 's';
            lines.appendChild(streak);
        }
    }

    /* ── Canvas particle network ─────────────────────────────── */
    function initCanvas() {
        const canvas = document.createElement('canvas');
        canvas.id = 'cyberCanvas';
        document.body.insertBefore(canvas, document.body.firstChild);

        const ctx = canvas.getContext('2d');
        let W, H;
        const PARTICLE_COUNT = 45;
        const CONNECTION_DIST = 140;
        const particles = [];

        function resize() {
            W = canvas.width = window.innerWidth;
            H = canvas.height = window.innerHeight;
        }

        class Particle {
            constructor() {
                this.reset();
            }
            reset() {
                this.x = Math.random() * W;
                this.y = Math.random() * H;
                this.vx = (Math.random() - 0.5) * 0.35;
                this.vy = (Math.random() - 0.5) * 0.35;
                this.r = 1 + Math.random() * 1.5;
                this.alpha = 0.15 + Math.random() * 0.25;
            }
            update() {
                this.x += this.vx;
                this.y += this.vy;
                if (this.x < 0 || this.x > W) this.vx *= -1;
                if (this.y < 0 || this.y > H) this.vy *= -1;
            }
            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(139, 92, 246, ${this.alpha})`;
                ctx.fill();
            }
        }

        function init() {
            resize();
            particles.length = 0;
            for (let i = 0; i < PARTICLE_COUNT; i++) {
                particles.push(new Particle());
            }
        }

        function drawConnections() {
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < CONNECTION_DIST) {
                        const alpha = (1 - dist / CONNECTION_DIST) * 0.08;
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }
        }

        function animate() {
            ctx.clearRect(0, 0, W, H);
            particles.forEach(p => { p.update(); p.draw(); });
            drawConnections();
            requestAnimationFrame(animate);
        }

        window.addEventListener('resize', () => {
            resize();
        });

        init();
        animate();
    }

    /* ── Boot ─────────────────────────────────────────────────── */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            injectCyberElements();
            initCanvas();
        });
    } else {
        injectCyberElements();
        initCanvas();
    }
})();
