'use strict';
(() => {
  const key = 'toumaii-theme';
  const root = document.documentElement;
  const apply = (choice) => {
    if (choice === 'light' || choice === 'dark') root.dataset.theme = choice;
    else delete root.dataset.theme;
  };

  try {
    apply(localStorage.getItem(key));
  } catch (error) {
    console.warn('Could not read the saved theme; using the device setting.', error);
  }

  document.addEventListener('DOMContentLoaded', () => {
    const control = document.getElementById('theme-choice');
    control.value = root.dataset.theme || 'system';
    control.addEventListener('change', () => {
      apply(control.value);
      try {
        if (control.value === 'system') localStorage.removeItem(key);
        else localStorage.setItem(key, control.value);
      } catch (error) {
        console.warn('Could not save the selected theme.', error);
        document.getElementById('announcement').textContent =
          'Theme changed for this visit, but the preference could not be saved.';
      }
    });
    control.closest('label').hidden = false;

    const header = control.closest('header');
    const updateOffset = () => {
      root.style.setProperty('--header-offset', `${header.offsetHeight + 16}px`);
    };
    updateOffset();
    new ResizeObserver(updateOffset).observe(header);
  });
})();
