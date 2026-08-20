const slides = [...document.querySelectorAll('.slide')];
const railLinks = [...document.querySelectorAll('.rail a')];
const progress = document.querySelector('.progress-bar');
const counter = document.querySelector('.counter');
const chapterButton = document.querySelector('[data-current-detail]');
let activeIndex = 0;

function setActive(index) {
  const safeIndex = Math.max(0, Math.min(slides.length - 1, index));
  activeIndex = safeIndex;
  railLinks.forEach((link, i) => link.classList.toggle('active', i === safeIndex));
  progress.style.width = `${((safeIndex + 1) / slides.length) * 100}%`;
  counter.textContent = `${String(safeIndex + 1).padStart(2, '0')} / ${String(slides.length).padStart(2, '0')}`;
  const detailTrigger = slides[safeIndex].querySelector('[data-dialog]');
  chapterButton.disabled = !detailTrigger;
  chapterButton.textContent = detailTrigger ? '展开本章' : '本章已完整';
}

chapterButton.addEventListener('click', () => {
  const trigger = slides[activeIndex].querySelector('[data-dialog]');
  if (trigger) trigger.click();
});

const observer = new IntersectionObserver((entries) => {
  const visible = entries
    .filter((entry) => entry.isIntersecting)
    .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
  if (visible) setActive(slides.indexOf(visible.target));
}, { threshold: [0.35, 0.55, 0.75] });
slides.forEach((slide) => observer.observe(slide));

const introVideo = document.querySelector('.intro-video');
if (introVideo) {
  const videoObserver = new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting && entry.intersectionRatio >= 0.55) {
      introVideo.play().catch(() => {});
    } else {
      introVideo.pause();
    }
  }, { threshold: [0, 0.55] });
  videoObserver.observe(introVideo);
}

document.querySelectorAll('[data-dialog]').forEach((trigger) => {
  trigger.addEventListener('click', () => {
    const dialog = document.getElementById(trigger.dataset.dialog);
    if (!dialog) return;
    dialog.showModal();
    document.body.classList.add('drawer-open');
  });
});

document.querySelectorAll('dialog').forEach((dialog) => {
  dialog.querySelectorAll('[data-close]').forEach((button) => {
    button.addEventListener('click', () => dialog.close());
  });
  dialog.addEventListener('close', () => document.body.classList.remove('drawer-open'));
  dialog.addEventListener('click', (event) => {
    const rect = dialog.getBoundingClientRect();
    const inside = event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
    if (!inside) dialog.close();
  });
});

const lightbox = document.getElementById('case-lightbox');
const lightboxImage = lightbox.querySelector('img');
const lightboxCaption = lightbox.querySelector('[data-caption]');
document.querySelectorAll('.case-card').forEach((card) => {
  card.addEventListener('click', () => {
    const image = card.querySelector('img');
    lightboxImage.src = image.src;
    lightboxImage.alt = image.alt;
    lightboxCaption.textContent = card.dataset.caption || image.alt;
    lightbox.showModal();
    document.body.classList.add('drawer-open');
  });
});

document.addEventListener('keydown', (event) => {
  if (document.querySelector('dialog[open]')) return;
  const current = railLinks.findIndex((link) => link.classList.contains('active'));
  if (event.key === 'ArrowDown' || event.key === 'PageDown') {
    event.preventDefault();
    slides[Math.min(slides.length - 1, current + 1)].scrollIntoView({ behavior: 'smooth' });
  }
  if (event.key === 'ArrowUp' || event.key === 'PageUp') {
    event.preventDefault();
    slides[Math.max(0, current - 1)].scrollIntoView({ behavior: 'smooth' });
  }
});

setActive(0);
