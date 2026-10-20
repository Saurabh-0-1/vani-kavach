const menuToggle = document.querySelector(".menu-toggle");
const siteNav = document.querySelector("#site-nav");
if (menuToggle && siteNav) {
  menuToggle.addEventListener("click", () => {
    const open = siteNav.classList.toggle("is-open");
    menuToggle.setAttribute("aria-expanded", open);
  });
  siteNav.addEventListener("click", (event) => {
    if (event.target.tagName === "A") siteNav.classList.remove("is-open");
  });
}

document.querySelectorAll("[data-password-toggle]").forEach((button) => {
  button.addEventListener("click", () => {
    const input = document.getElementById(button.dataset.passwordToggle);
    const visible = input.type === "text";
    input.type = visible ? "password" : "text";
    button.textContent = visible ? "Show" : "Hide";
    button.setAttribute("aria-label", visible ? "Show password" : "Hide password");
  });
});

const validEmail = (value) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
document.querySelectorAll("[data-form-kind]").forEach((form) => {
  const message = form.querySelector(".form-message");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const kind = form.dataset.formKind;
    const email = form.querySelector('input[type="email"]');
    const password = form.querySelector('input[name="password"]');
    const confirm = form.querySelector('input[name="confirmPassword"]');
    const name = form.querySelector('input[name="fullName"]');
    let error = "";
    if (name && !name.value.trim()) error = "Enter your full name.";
    else if (email && !email.value.trim()) error = "Enter your email address.";
    else if (email && !validEmail(email.value)) error = "Enter a valid email address.";
    else if (password && !password.value) error = "Enter your password.";
    else if (kind === "signup" && password.value.length < 8) error = "Password must be at least 8 characters.";
    else if (confirm && password.value !== confirm.value) error = "Passwords do not match.";
    if (error) {
      message.textContent = error;
      message.className = "form-message error";
      return;
    }
    message.textContent = kind === "forgot"
      ? "Demo only: no reset email was sent because a password-recovery backend is not configured."
      : "Form validated, but account authentication is not available until a secure backend is implemented.";
    message.className = "form-message notice";
  });
});
