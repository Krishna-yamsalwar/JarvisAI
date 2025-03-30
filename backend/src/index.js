function stripHtml(html) {
    let tempDiv = document.createElement("div");
    tempDiv.innerHTML = html;
    return tempDiv.textContent || tempDiv.innerText || "";
}
 let emails = []
  async function fetchEmails() {
    try {
        const response = await fetch("http://127.0.0.1:5000/emails");
         emails = await response.json();
         emails=emails.sort((a, b) =>  b.id-a.id);
        console.log(emails)
        renderEmailList();
        

    } catch (error) {
        console.error("Error fetching emails:", error);
    }
}

console.log(emails)

// DOM Elements
const menuToggle = document.getElementById('menuToggle');
const sidebar = document.getElementById('sidebar');
const emailItems = document.getElementById('emailItems');
const emailDetail = document.getElementById('emailDetail');

// Toggle mobile menu
menuToggle.addEventListener('click', () => {
    sidebar.classList.toggle('active');
});

// Close sidebar when clicking outside on mobile
document.addEventListener('click', (e) => {
    if (window.innerWidth <= 1024) {
        if (!sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
            sidebar.classList.remove('active');
        }
    }
});

// Toggle reply window
function toggleReplyWindow() {
    const replyWindow = document.querySelector('.reply-window');
    replyWindow.classList.toggle('active');
}

// Handle reply submission
function handleReply(event) {
    event.preventDefault();
    const textarea = document.querySelector('.reply-textarea');
    console.log('Reply sent:', textarea.value);
    textarea.value = '';
    toggleReplyWindow();
}

// Render email list
function renderEmailList() {
  
    emailItems.innerHTML = emails.map(email => `
        <div class="email-item ${email.read ? '' : 'unread'}" data-id="${email.id}">
            <div class="email-item-header">
                <span class="email-from">${email.from}</span>
                <span class="email-date">${email.date}</span>
            </div>
            <div class="email-subject">${email.subject}</div>
            <div class="email-preview">${ stripHtml(email.body)}</div>
        </div>
    `).join('');

    // Add click handlers to email items
    document.querySelectorAll('.email-item').forEach(item => {
        item.addEventListener('click', () => {
            const emailId = parseInt(item.dataset.id);
            showEmailDetail(emailId);
            
            // Update active state
            document.querySelectorAll('.email-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            
            // Show detail view on mobile
            if (window.innerWidth <= 1024) {
                emailDetail.classList.add('active');
            }
        });
    });
    console.log('email_rendered')
}

// Show email detail
function showEmailDetail(emailId) {
    console.log('called showEmail')
    const email = emails.find(e => e.id === emailId);
    if (!email) return;

    const detailContent = `
        <div class="email-content">
            ${window.innerWidth <= 1024 ? `
                <button class="back-button" onclick="hideEmailDetail()">
                    ← Back to list
                </button>
            ` : ''}
            <div class="email-content-header">
                <h1 class="email-content-subject">${email.subject}</h1>
                <div class="email-content-meta">
                    <span>${email.from}</span>
                    <span>${email.date}</span>
                </div>
            </div>
            <div class="email-content-body">
                ${email.body.replace(/\n/g, '<br>')}
            </div>
            <button class="reply-button" onclick="toggleReplyWindow()">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="9 14 4 9 9 4"></polyline>
                    <path d="M20 20v-7a4 4 0 0 0-4-4H4"></path>
                </svg>
                Reply
            </button>
            <div class="reply-window">
                <form id="replyForm">
<input class="reply-header" placeholder="Enter email" value="${email.sender_email}" type="email" name="email" required />
<textarea class="reply-textarea" name="body" placeholder="Write your reply here..." required></textarea>
<div class="reply-actions">
<button class="reply-cancel" type="button" onclick="toggleReplyWindow()">Cancel</button>
<button type="button" class="generate-reply"   onclick="generateReply()">Generate Reply</button>

<button class="reply-send" type="submit">Send Reply</button>
</div>
</form>

<!-- Success/Error Message (Hidden Initially) -->
<div id="responseMessage" style="display: none;"></div>
            </div>
        </div>
    `;


    emailDetail.innerHTML = detailContent;
    email.read = true;
}

// Hide email detail (mobile)
function hideEmailDetail() {
    emailDetail.classList.remove('active');
}
fetchEmails()