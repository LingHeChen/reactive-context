/* GSAP is a progressive enhancement: missing scripts leave all content readable. */
(() => {
  'use strict';
  const { gsap, ScrollTrigger } = window;
  if (!gsap || !ScrollTrigger) return;
  gsap.registerPlugin(ScrollTrigger);
  ScrollTrigger.config({ ignoreMobileResize: true });

  const chapters = [...document.querySelectorAll('.chapter')];
  const navLinks = [...document.querySelectorAll('.nav nav a')];
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const progressed = new WeakSet();
  const revealTweens = new Map();
  let motionContext = null;
  let responsive = null;
  let refreshTimer;

  // A reading indicator, not a replacement for the native scrollbar.
  const progress = document.querySelector('.reading-progress span');
  ScrollTrigger.create({
    start: 0,
    end: () => Math.max(1, ScrollTrigger.maxScroll(window)),
    onUpdate: (self) => {
      progress.style.transform = `scaleX(${self.progress})`;
      const active = [...navLinks].reverse().find(link => document.querySelector(link.hash).getBoundingClientRect().top <= innerHeight * .4);
      updateNav(active?.hash.slice(1) || '');
    },
    onRefresh: (self) => { progress.style.transform = `scaleX(${self.progress})`; },
  });

  function updateNav(id) {
    navLinks.forEach((link) => {
      if (link.hash === `#${id}`) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }
  chapters.forEach((chapter, index) => {
    const ink = document.createElement('i');
    ink.className = 'chapter-ink';
    ink.setAttribute('aria-hidden', 'true');
    chapter.querySelector('.chapter-label').append(ink);
    ScrollTrigger.create({
      trigger: chapter,
      start: 'top 40%',
      end: 'bottom 40%',

    });
  });

  function reveal(element, delay = 0, distance = 22) {
    if (!element || progressed.has(element)) return;
    // Preserve in-view content on reload, deep links, resize and re-enabling motion.
    if (element.getBoundingClientRect().top < innerHeight * .88) {
      progressed.add(element);
      return;
    }
    const tween = gsap.fromTo(element,
      { y: distance, opacity: .25 },
      {
        y: 0, opacity: 1, duration: .75, delay, ease: 'power3.out',
        scrollTrigger: { trigger: element, start: 'top 92%', once: true },
        onComplete: () => {
          progressed.add(element);
          revealTweens.delete(element);
          gsap.set(element, { clearProps: 'opacity,transform' });
        },
      },
    );
    revealTweens.set(element, tween);
  }

  function buildMotion() {
    responsive?.revert();
    responsive = null;
    motionContext?.revert();
    motionContext = null;
    revealTweens.clear();
    if (reduced.matches || document.documentElement.classList.contains('motion-paused')) return;

    motionContext = gsap.context(() => {
      chapters.forEach((chapter) => {
        reveal(chapter.querySelector('.editorial-heading h2'));
        reveal(chapter.querySelector('.editorial-heading .intro-text'), .1, 16);
        gsap.fromTo(chapter.querySelector('.chapter-ink'), { scaleX: 0 }, {
          scaleX: 1, ease: 'none',
          scrollTrigger: { trigger: chapter, start: 'top 88%', end: 'top 26%', scrub: .35 },
        });
      });
      ['.problem-list article', '.memory-types article', '.candidate-flow li', '.decision-table article', '.evidence-grid article'].forEach((selector) => {
        document.querySelectorAll(selector).forEach((element, index) => reveal(element, (index % 3) * .07, 18));
      });

      reveal(document.querySelector('.start-copy'), 0, 20);
      reveal(document.querySelector('.code-window'), .12, 24);

      const map = document.querySelector('.architecture-map');
      const timeline = gsap.timeline({
        scrollTrigger: { trigger: map, start: 'top 82%', end: 'bottom 78%', scrub: .4 },
        defaults: { ease: 'power1.out', duration: 1 },
      });
      timeline.fromTo('.map-input', { opacity: .4, y: 12 }, { opacity: 1, y: 0 })
        .fromTo('.map-sources', { opacity: .35, x: -12 }, { opacity: 1, x: 0 }, '-=.6')
        .fromTo('.map-core > div', { opacity: .35, y: 16 }, { opacity: 1, y: 0, stagger: .4 }, '-=.3')
        .fromTo('.map-storage', { opacity: .35, x: 12 }, { opacity: 1, x: 0 }, '-=.8')
        .fromTo('.map-output', { opacity: .35, y: 12 }, { opacity: 1, y: 0 }, '-=.4');
    });

    // Mobile keeps the reveals but skips large background and cover transforms.
    responsive = gsap.matchMedia();
    responsive.add('(min-width: 801px) and (prefers-reduced-motion: no-preference)', () => {
      gsap.to('.hero-cover', {
        scale: .975, y: 24, ease: 'none',
        scrollTrigger: { trigger: '.hero', start: 'top top', end: 'bottom top', scrub: .5 },
      });
      gsap.fromTo('.memory-cover-text', { y: 22 }, {
        y: -22, ease: 'none',
        scrollTrigger: { trigger: '.memory-cover', start: 'top bottom', end: 'bottom top', scrub: .5 },
      });
    });
    ScrollTrigger.refresh();
  }

  // Focusing a control immediately reveals its animated ancestors.
  document.addEventListener('focusin', (event) => {
    for (const [element, tween] of revealTweens) {
      if (element.contains(event.target)) tween.progress(1);
    }
  });
  document.addEventListener('context-motion-change', buildMotion);

  // Keep the complete panel visible while native scrolling selects its snapshots.
  const studio = document.querySelector('.studio');
  let demoTrigger = null;
  let demoMode = false;
  function syncDemoMode() {
    if (!window.contextDemo?.ready) return;
    const fits = innerWidth >= 1000 && innerHeight >= studio.offsetHeight + 124;
    const enabled = fits && !reduced.matches && !document.documentElement.classList.contains('motion-paused');
    if (enabled === demoMode) return;
    demoMode = enabled;
    demoTrigger?.kill();
    demoTrigger = null;
    window.contextScrollDemo = null;
    studio.classList.toggle('scroll-driven', enabled);
    if (enabled) {
      window.contextDemo.stop();
      demoTrigger = ScrollTrigger.create({
        id: 'demo-stages', trigger: studio, pin: true, refreshPriority: 1,
        start: 'top 108px', end: () => `+=${innerHeight * 2.4}`,
        invalidateOnRefresh: true,
        onUpdate(self) {
          window.contextDemo.setStep(Math.min(3, Math.floor(self.progress * 4)));
          studio.style.setProperty('--demo-progress', self.progress);
        },
      });
      window.contextScrollDemo = { restart() { scrollTo({ top: demoTrigger.start + 1, behavior: 'smooth' }); } };
    }
    ScrollTrigger.sort();
    window.contextDemo.stop();
    document.querySelector('#replay-toggle').setAttribute('aria-label', enabled ? '滚动回到第一个阶段' : '自动演示四个阶段');
    ScrollTrigger.refresh();
  }
  document.addEventListener('context-demo-select', ({ detail }) => {
    if (demoTrigger) scrollTo({ top: demoTrigger.start + (detail.index + .5) / 4 * (demoTrigger.end - demoTrigger.start), behavior: 'instant' });
  });
  document.addEventListener('context-demo-ready', syncDemoMode);
  document.addEventListener('context-motion-change', syncDemoMode);
  window.addEventListener('resize', syncDemoMode);
  document.fonts.ready.then(syncDemoMode);
  syncDemoMode();

  function refreshAfterLayout() {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => { syncDemoMode(); ScrollTrigger.refresh(); }, 100);
  }
  document.querySelectorAll('details').forEach((detail) => detail.addEventListener('toggle', refreshAfterLayout));
  // Replay output and disclosures can change document height after initial measurement.
  new ResizeObserver(refreshAfterLayout).observe(document.querySelector('main'));
  document.fonts.ready.then(refreshAfterLayout);
  window.addEventListener('pageshow', refreshAfterLayout);
  buildMotion();
})();
