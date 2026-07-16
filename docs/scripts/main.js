const revealObserver = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      }
    }
  },
  { threshold: 0.12 }
);

document.querySelectorAll(".reveal").forEach((node) => {
  revealObserver.observe(node);
});

document.querySelectorAll("[data-carousel]").forEach((carousel) => {
  const viewport = carousel.querySelector(".carousel-viewport");
  const track = carousel.querySelector(".carousel-track");
  const initialSlides = Array.from(carousel.querySelectorAll(".carousel-slide"));
  const prevButton = carousel.querySelector('[data-carousel-control="prev"]');
  const nextButton = carousel.querySelector('[data-carousel-control="next"]');
  const currentNode = carousel.querySelector("[data-carousel-current]");
  const totalNode = carousel.querySelector("[data-carousel-total]");
  const autoplayDelay = Number(carousel.dataset.autoplay || 5600);
  const transitionStyle = "transform 560ms cubic-bezier(0.22, 1, 0.36, 1)";

  if (!track || initialSlides.length === 0 || !viewport) {
    return;
  }

  const logicalTotal = initialSlides.length;
  let slides = [];
  let logicalIndex = 0;
  let autoplayId = null;
  let isAnimating = false;
  let dragState = null;

  const formatIndex = (value) => String(value).padStart(2, "0");
  const steadyDomIndex = logicalTotal > 1 ? 1 : 0;

  const refreshSlides = () => {
    slides = Array.from(track.querySelectorAll(".carousel-slide"));
  };

  if (logicalTotal === 2) {
    initialSlides.forEach((slide) => {
      track.appendChild(slide.cloneNode(true));
    });
  }

  refreshSlides();

  if (logicalTotal > 1) {
    track.insertBefore(track.lastElementChild, track.firstElementChild);
    refreshSlides();
  }

  const updateCounter = () => {
    if (currentNode) currentNode.textContent = formatIndex(logicalIndex + 1);
    if (totalNode) totalNode.textContent = formatIndex(logicalTotal);
  };

  const stopAutoplay = () => {
    if (autoplayId !== null) {
      window.clearInterval(autoplayId);
      autoplayId = null;
    }
  };

  const setTransition = (enabled) => {
    track.style.transition = enabled ? transitionStyle : "none";
  };

  const getCenterOffset = (domIndex) => {
    const activeSlide = slides[domIndex];
    if (!activeSlide) return 0;

    return viewport.clientWidth / 2 - (activeSlide.offsetLeft + activeSlide.clientWidth / 2);
  };

  const updateSlideClasses = (activeDomIndex) => {
    slides.forEach((slide, slideIndex) => {
      slide.classList.toggle("is-active", slideIndex === activeDomIndex);
      slide.classList.toggle("is-neighbor", Math.abs(slideIndex - activeDomIndex) === 1);
    });
  };

  const snapToSteady = () => {
    refreshSlides();
    setTransition(false);
    updateSlideClasses(steadyDomIndex);
    track.style.transform = `translate3d(${getCenterOffset(steadyDomIndex)}px, 0, 0)`;
    void track.offsetWidth;
    setTransition(true);
    updateCounter();
  };

  const animateTo = (domIndex) => {
    updateSlideClasses(domIndex);
    track.style.transform = `translate3d(${getCenterOffset(domIndex)}px, 0, 0)`;
  };

  const finishMove = (direction) => {
    if (direction > 0) {
      track.appendChild(track.firstElementChild);
      logicalIndex = (logicalIndex + 1) % logicalTotal;
    } else {
      track.insertBefore(track.lastElementChild, track.firstElementChild);
      logicalIndex = (logicalIndex - 1 + logicalTotal) % logicalTotal;
    }

    snapToSteady();
    isAnimating = false;
  };

  const move = (direction) => {
    if (logicalTotal < 2 || isAnimating) {
      return;
    }

    isAnimating = true;
    refreshSlides();
    const targetDomIndex = direction > 0 ? steadyDomIndex + 1 : steadyDomIndex - 1;
    animateTo(targetDomIndex);

    window.setTimeout(() => {
      finishMove(direction);
    }, 580);
  };

  const startAutoplay = () => {
    if (logicalTotal < 2 || autoplayId !== null) {
      return;
    }

    autoplayId = window.setInterval(() => {
      move(1);
    }, autoplayDelay);
  };

  const restartAutoplay = () => {
    stopAutoplay();
    startAutoplay();
  };

  const getDragThreshold = () => {
    const activeSlide = slides[steadyDomIndex];
    if (!activeSlide) return 56;
    return Math.max(56, activeSlide.clientWidth * 0.12);
  };

  const startDrag = (event) => {
    if (logicalTotal < 2 || isAnimating) {
      return;
    }

    dragState = {
      pointerId: event.pointerId,
      startX: event.clientX,
      deltaX: 0,
      baseOffset: getCenterOffset(steadyDomIndex),
    };

    stopAutoplay();
    carousel.classList.add("is-dragging");
    viewport.setPointerCapture?.(event.pointerId);
    setTransition(false);
  };

  const onDrag = (event) => {
    if (!dragState || event.pointerId !== dragState.pointerId) {
      return;
    }

    dragState.deltaX = event.clientX - dragState.startX;
    track.style.transform = `translate3d(${dragState.baseOffset + dragState.deltaX}px, 0, 0)`;
  };

  const endDrag = (event) => {
    if (!dragState || event.pointerId !== dragState.pointerId) {
      return;
    }

    viewport.releasePointerCapture?.(event.pointerId);
    const { deltaX } = dragState;
    dragState = null;
    carousel.classList.remove("is-dragging");
    setTransition(true);

    if (Math.abs(deltaX) > getDragThreshold()) {
      move(deltaX < 0 ? 1 : -1);
    } else {
      snapToSteady();
    }

    startAutoplay();
  };

  if (logicalTotal < 2) {
    carousel.classList.add("is-static");
    prevButton?.setAttribute("disabled", "disabled");
    nextButton?.setAttribute("disabled", "disabled");
  }

  prevButton?.addEventListener("click", () => {
    move(-1);
    restartAutoplay();
  });

  nextButton?.addEventListener("click", () => {
    move(1);
    restartAutoplay();
  });

  viewport.addEventListener("pointerdown", startDrag);
  viewport.addEventListener("pointermove", onDrag);
  viewport.addEventListener("pointerup", endDrag);
  viewport.addEventListener("pointercancel", endDrag);
  viewport.addEventListener("pointerleave", endDrag);

  carousel.addEventListener("focusin", stopAutoplay);
  carousel.addEventListener("focusout", startAutoplay);

  window.addEventListener("resize", snapToSteady);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopAutoplay();
    } else {
      startAutoplay();
    }
  });

  snapToSteady();
  startAutoplay();
});

const copyButton = document.querySelector("[data-copy-target]");
if (copyButton) {
  copyButton.addEventListener("click", async () => {
    const target = document.getElementById(copyButton.dataset.copyTarget);
    if (!target) return;

    try {
      await navigator.clipboard.writeText(target.textContent);
      const previous = copyButton.textContent;
      copyButton.textContent = "Copied";
      window.setTimeout(() => {
        copyButton.textContent = previous;
      }, 1400);
    } catch (error) {
      copyButton.textContent = "Copy failed";
      window.setTimeout(() => {
        copyButton.textContent = "Copy";
      }, 1400);
    }
  });
}
