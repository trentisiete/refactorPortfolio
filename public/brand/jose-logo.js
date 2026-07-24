(() => {
  "use strict";

  const TOTAL_DURATION = 3100;
  const DRAW_END = 900;
  const MORPH_START = 1050;
  const MORPH_END = 1800;

  /*
   * Todos los estados contienen exactamente:
   * M + seis curvas cúbicas C.
   *
   * El path comienza en la parte superior de la J. Durante el morphing,
   * su extremo izquierdo asciende hasta encontrarse con ese punto y cerrar
   * el pétalo. La base del pétalo final queda exactamente en 0,0.
   */
  const PATH_STRINGS = {
    j: `
      M 18 -155
      C 18 -130 18 -100 18 -70
      C 18 -40 18 -10 15 10
      C 12 32 -2 45 -23 45
      C -42 45 -55 34 -58 18
      C -60 10 -60 3 -60 -4
      C -60 -4 -60 -4 -60 -4
      C -60 -4 -60 -4 -60 -4
    `,
    curling: `
      M 16 -155
      C 20 -130 22 -100 20 -72
      C 18 -42 10 -12 2 6
      C -8 25 -23 30 -36 24
      C -51 18 -58 0 -54 -24
      C -49 -58 -34 -102 -18 -132
      C -12 -143 -6 -151 -2 -154
    `,
    almostClosed: `
      M 8 -158
      C 20 -148 28 -126 25 -96
      C 21 -52 5 -17 0 0
      C -7 -18 -23 -52 -26 -92
      C -29 -124 -23 -145 -10 -154
      C -6 -157 -3 -159 0 -159
      C 3 -159 6 -159 8 -158
    `,
    petal: `
      M 0 -160
      C 18 -155 29 -135 26 -100
      C 22 -55 4 -18 0 0
      C -4 -18 -22 -55 -26 -100
      C -29 -135 -18 -155 0 -160
      C 0 -160 0 -160 0 -160
      C 0 -160 0 -160 0 -160
    `,
  };

  const MORPH_KEYFRAMES = [
    { at: 0, values: pathNumbers(PATH_STRINGS.j) },
    { at: 0.4, values: pathNumbers(PATH_STRINGS.curling) },
    { at: 0.72, values: pathNumbers(PATH_STRINGS.almostClosed) },
    { at: 1, values: pathNumbers(PATH_STRINGS.petal) },
  ];

  const COPY_SCHEDULE = {
    90: { start: 1950, duration: 240 },
    180: { start: 2030, duration: 240 },
    270: { start: 2110, duration: 240 },
    45: { start: 2350, duration: 230 },
    135: { start: 2440, duration: 230 },
    225: { start: 2530, duration: 230 },
    315: { start: 2620, duration: 230 },
  };

  const controllers = [];
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  function pathNumbers(path) {
    return (path.match(/-?\d+(?:\.\d+)?/g) || []).map(Number);
  }

  function number(value) {
    const rounded = Math.round(value * 1000) / 1000;
    return Object.is(rounded, -0) ? "0" : String(rounded);
  }

  function pathFromNumbers(values) {
    let path = `M ${number(values[0])} ${number(values[1])}`;

    for (let index = 2; index < values.length; index += 6) {
      path += [
        " C ",
        number(values[index]),
        " ",
        number(values[index + 1]),
        " ",
        number(values[index + 2]),
        " ",
        number(values[index + 3]),
        " ",
        number(values[index + 4]),
        " ",
        number(values[index + 5]),
      ].join("");
    }

    return path;
  }

  function clamp(value, minimum = 0, maximum = 1) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function mix(from, to, progress) {
    return from + (to - from) * progress;
  }

  function easeInOutCubic(progress) {
    return progress < 0.5
      ? 4 * progress * progress * progress
      : 1 - Math.pow(-2 * progress + 2, 3) / 2;
  }

  function easeOutCubic(progress) {
    return 1 - Math.pow(1 - progress, 3);
  }

  function interpolateValues(from, to, progress) {
    return from.map((value, index) => mix(value, to[index], progress));
  }

  function morphValues(progress) {
    const bounded = clamp(progress);

    for (let index = 0; index < MORPH_KEYFRAMES.length - 1; index += 1) {
      const current = MORPH_KEYFRAMES[index];
      const next = MORPH_KEYFRAMES[index + 1];

      if (bounded <= next.at) {
        const localProgress = (bounded - current.at) / (next.at - current.at);
        return interpolateValues(
          current.values,
          next.values,
          easeInOutCubic(clamp(localProgress)),
        );
      }
    }

    return MORPH_KEYFRAMES[MORPH_KEYFRAMES.length - 1].values;
  }

  function copyGrowth(progress) {
    const bounded = clamp(progress);
    const rotationProgress = easeOutCubic(bounded);
    let scale;

    if (bounded < 0.78) {
      scale = mix(0.05, 1.06, easeOutCubic(bounded / 0.78));
    } else {
      scale = mix(1.06, 1, easeInOutCubic((bounded - 0.78) / 0.22));
    }

    return {
      opacity: easeOutCubic(clamp(bounded * 1.8)),
      rotationProgress,
      scale,
    };
  }

  class JoseLogoController {
    constructor(svg, index) {
      this.svg = svg;
      this.master = svg.querySelector("[data-logo-master]");
      this.original = svg.querySelector("[data-logo-original]");
      this.copies = Array.from(svg.querySelectorAll(".jose-logo__copy"));
      this.frame = 0;
      this.startedAt = 0;

      if (!this.master || !this.original) {
        throw new Error("El SVG del logo no contiene los elementos requeridos.");
      }

      /*
       * Evita colisiones de IDs si se insertan varias instancias del logo
       * en una misma página.
       */
      const oldId = this.master.id;
      const uniqueId = `${oldId}-${index + 1}`;
      this.master.id = uniqueId;
      svg.querySelectorAll("use").forEach((use) => {
        if (use.getAttribute("href") === `#${oldId}`) {
          use.setAttribute("href", `#${uniqueId}`);
        }
      });

      this.reset();
    }

    cancel() {
      if (this.frame) {
        window.cancelAnimationFrame(this.frame);
        this.frame = 0;
      }
    }

    reset() {
      this.cancel();
      this.master.setAttribute("d", pathFromNumbers(MORPH_KEYFRAMES[0].values));

      const length = this.master.getTotalLength();
      this.original.style.strokeDasharray = `${length} ${length}`;
      this.original.style.strokeDashoffset = String(length);

      this.copies.forEach((copy) => {
        const angle = Number(copy.dataset.angle);
        copy.style.opacity = "0";
        copy.setAttribute("transform", `rotate(${angle - 8}) scale(0.05)`);
      });

      this.svg.dataset.logoState = "reset";
    }

    showFinal() {
      this.cancel();
      this.master.setAttribute(
        "d",
        pathFromNumbers(MORPH_KEYFRAMES[MORPH_KEYFRAMES.length - 1].values),
      );
      this.original.style.strokeDasharray = "none";
      this.original.style.strokeDashoffset = "0";

      this.copies.forEach((copy) => {
        const angle = Number(copy.dataset.angle);
        copy.style.opacity = "1";
        copy.setAttribute("transform", `rotate(${angle}) scale(1)`);
      });

      this.svg.dataset.logoState = "final";
    }

    renderCopy(copy, time) {
      const angle = Number(copy.dataset.angle);
      const schedule = COPY_SCHEDULE[angle];
      const progress = clamp((time - schedule.start) / schedule.duration);

      if (time < schedule.start) {
        copy.style.opacity = "0";
        copy.setAttribute("transform", `rotate(${angle - 8}) scale(0.05)`);
        return;
      }

      const growth = copyGrowth(progress);
      const rotation = mix(angle - 8, angle, growth.rotationProgress);

      copy.style.opacity = String(growth.opacity);
      copy.setAttribute(
        "transform",
        `rotate(${number(rotation)}) scale(${number(growth.scale)})`,
      );
    }

    render(time) {
      const boundedTime = Math.min(time, TOTAL_DURATION);

      if (boundedTime <= DRAW_END) {
        const progress = easeInOutCubic(boundedTime / DRAW_END);
        const length = this.master.getTotalLength();
        this.original.style.strokeDashoffset = String(length * (1 - progress));
      } else {
        /*
         * El pétalo final es más largo que la J. Retiramos el patrón de
         * guiones al terminar el dibujo para que el morphing nunca genere
         * un hueco accidental en el contorno.
         */
        this.original.style.strokeDasharray = "none";
        this.original.style.strokeDashoffset = "0";
      }

      if (boundedTime < MORPH_START) {
        this.master.setAttribute("d", pathFromNumbers(MORPH_KEYFRAMES[0].values));
      } else if (boundedTime < MORPH_END) {
        const progress = (boundedTime - MORPH_START) / (MORPH_END - MORPH_START);
        this.master.setAttribute("d", pathFromNumbers(morphValues(progress)));
      } else {
        this.master.setAttribute(
          "d",
          pathFromNumbers(MORPH_KEYFRAMES[MORPH_KEYFRAMES.length - 1].values),
        );
      }

      this.copies.forEach((copy) => this.renderCopy(copy, boundedTime));
    }

    play() {
      if (reducedMotion.matches) {
        this.showFinal();
        return;
      }

      this.reset();
      this.svg.dataset.logoState = "playing";
      this.startedAt = performance.now();

      const tick = (now) => {
        const elapsed = now - this.startedAt;
        this.render(elapsed);

        if (elapsed < TOTAL_DURATION) {
          this.frame = window.requestAnimationFrame(tick);
        } else {
          this.showFinal();
        }
      };

      this.frame = window.requestAnimationFrame(tick);
    }
  }

  function eachController(action) {
    controllers.forEach((controller) => controller[action]());
  }

  window.playJoseLogo = () => eachController("play");
  window.resetJoseLogo = () => eachController("reset");
  window.showFinalJoseLogo = () => eachController("showFinal");

  function initialize() {
    document.querySelectorAll("[data-jose-logo]").forEach((svg, index) => {
      controllers.push(new JoseLogoController(svg, index));
    });

    document.querySelectorAll("[data-logo-action]").forEach((button) => {
      button.addEventListener("click", () => {
        const action = button.dataset.logoAction;

        if (action === "play") window.playJoseLogo();
        if (action === "reset") window.resetJoseLogo();
        if (action === "final") window.showFinalJoseLogo();
      });
    });

    if (reducedMotion.matches) {
      window.showFinalJoseLogo();
    } else {
      window.playJoseLogo();
    }

    reducedMotion.addEventListener("change", (event) => {
      if (event.matches) {
        window.showFinalJoseLogo();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
