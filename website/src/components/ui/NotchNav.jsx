import React, { useState, useEffect, useRef, useMemo, useCallback, useLayoutEffect } from "react";
import "./NotchNav.css";

export function NotchNav({
  items = [],
  value,
  defaultValue,
  onValueChange,
  ariaLabel = "Primary",
  className = "",
}) {
  const isControlled = value !== undefined;
  const [active, setActive] = useState(value ?? defaultValue ?? items[0]?.value ?? "");
  const [ready, setReady] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    if (isControlled && value !== undefined) setActive(value);
  }, [isControlled, value]);

  const containerRef = useRef(null);
  const itemRefs = useRef([]);
  const [notchRect, setNotchRect] = useState(null);

  const activeIndex = useMemo(
    () => Math.max(0, items.findIndex((i) => i.value === active)),
    [items, active]
  );

  const updateNotch = useCallback(() => {
    const c = containerRef.current;
    const el = itemRefs.current[activeIndex];
    if (!c || !el) return;
    const cRect = c.getBoundingClientRect();
    const eRect = el.getBoundingClientRect();
    const left = eRect.left - cRect.left;
    const width = eRect.width;
    setNotchRect({ left, width });
    setReady(true);
  }, [activeIndex]);

  useLayoutEffect(() => {
    updateNotch();
    const onResize = () => updateNotch();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [updateNotch]);

  const focusItem = (index) => {
    const el = itemRefs.current[Math.max(0, Math.min(items.length - 1, index))];
    el?.focus();
  };

  const commitChange = (next) => {
    if (!isControlled) setActive(next);
    onValueChange?.(next);
  };

  useEffect(() => {
    const mql = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReducedMotion(mql.matches);
    onChange();
    mql.addEventListener?.("change", onChange);
    return () => mql.removeEventListener?.("change", onChange);
  }, []);

  return (
    <nav aria-label={ariaLabel} className={`notch-nav-root ${className}`}>
      <div ref={containerRef} className="notch-container">
        {/* Items */}
        <ul
          role="menubar"
          className="notch-menu"
          onKeyDown={(e) => {
            const key = e.key;
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(key)) return;
            e.preventDefault();
            if (key === "ArrowRight") focusItem(activeIndex + 1);
            if (key === "ArrowLeft") focusItem(activeIndex - 1);
            if (key === "Home") focusItem(0);
            if (key === "End") focusItem(items.length - 1);
          }}
        >
          {items.map((item, idx) => {
            const isActive = item.value === active;
            return (
              <li key={item.value} role="none">
                <button
                  ref={(el) => (itemRefs.current[idx] = el)}
                  role="menuitem"
                  aria-current={isActive ? "page" : undefined}
                  aria-pressed={isActive || undefined}
                  tabIndex={isActive ? 0 : -1}
                  onClick={() => {
                    commitChange(item.value);
                    if (item.href) {
                      const el = document.querySelector(item.href);
                      if (el) el.scrollIntoView({ behavior: 'smooth' });
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      commitChange(item.value);
                    }
                  }}
                  className={`notch-btn ${isActive ? 'active' : ''}`}
                >
                  <span>{item.label}</span>
                </button>
              </li>
            );
          })}
        </ul>

        {/* Notch indicator (SVG) */}
        {notchRect && (
          <div
            aria-hidden="true"
            className={`notch-indicator ${ready ? 'ready' : ''} ${reducedMotion ? 'reduced-motion' : ''}`}
            style={{
              transform: `translate3d(${notchRect.left}px, 0, 0)`,
              width: notchRect.width,
            }}
          >
            <svg
              width="100%"
              height="100%"
              viewBox="0 0 100 20"
              preserveAspectRatio="none"
            >
              <path
                d="M 2 1 H 98 Q 99 1 99 2 V 10 H 88 Q 87.2 10 86.6 11.4 L 84.8 18 H 15.2 L 13.4 11.4 Q 12.8 10 12 10 H 2 Q 1 10 1 9 V 2 Q 1 1 2 1 Z"
                fill="currentColor"
              />
            </svg>
          </div>
        )}
      </div>
    </nav>
  );
}
