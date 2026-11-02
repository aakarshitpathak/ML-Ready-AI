// "Show password" checkbox: flips every password field in the same form
// between hidden and visible.
function togglePasswords(checkbox) {
    var form = checkbox.closest('form');
    if (!form) return;
    form.querySelectorAll('input[name="password"], input[name="confirm_password"]').forEach(function (input) {
        input.type = checkbox.checked ? 'text' : 'password';
    });
}
