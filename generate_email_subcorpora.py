import os
import sys
import io
import random
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Seed for deterministic generation
random.seed(42)
np.random.seed(42)

def generate_augmented_email_corpus():
    output_dir = os.path.join(os.path.dirname(__file__), 'data', 'Email_dataset')
    os.makedirs(output_dir, exist_ok=True)
    
    records = []

    # =========================================================================
    # 1. CATEGORY A: BENIGN CONVERSATIONAL & PERSONAL EMAILS (Label 0)
    # =========================================================================
    greetings = ["Hi", "Hello", "Hey", "Good morning", "Good afternoon", "Dear", "Hey there"]
    names = ["Alex", "Sam", "Sugumaran", "Jordan", "Taylor", "Chris", "Morgan", "Naveen", "Priya", "David", "Emma", "Daniel", "Sarah", "Michael"]
    closings = ["Best regards", "Thanks", "Cheers", "Warmly", "See you soon", "Talk later", "Best", "Sincerely", "Have a great day"]

    conversational_templates = [
        "Hi how are you?",
        "Hi how are you doing today?",
        "Hello, hope you are having a wonderful week.",
        "Hey {name}, how are things going with the project?",
        "Thanks for your message! Let's catch up tomorrow afternoon.",
        "See you tomorrow at the office. Don't forget to bring the notes.",
        "Congratulations on your new role! So excited for you.",
        "Great job on the presentation today, everyone was impressed.",
        "Are you free for a quick coffee chat this Thursday?",
        "Just checking in to see if you received my previous note.",
        "Thanks for the help earlier today, really appreciate it!",
        "Let's reschedule our one-on-one meeting to next Monday at 2 PM.",
        "Hope you had a relaxing weekend! Let me know when you have 5 minutes to talk.",
        "Can you send over the slides whenever you get a chance? No rush.",
        "Thanks for introducing me to {name}. We had a productive conversation.",
        "I'll be out of office tomorrow for a doctor's appointment. Back on Friday.",
        "Happy birthday {name}! Wishing you all the best.",
        "Quick reminder about our lunch plan today at 1 PM at the cafeteria.",
        "Sounds good to me, let's proceed with that plan.",
        "I reviewed the draft you shared and added a few comments. Looks great overall!",
        "Let's sync up after the team standup to finalize the agenda.",
        "Could you please review the attached meeting minutes when you have time?",
        "Thank you so much for organizing the team outing, it was a lot of fun.",
        "I will be working from home today. Reachable via chat and email.",
        "Great catching up yesterday. Looking forward to our next collaboration.",
        "Please let me know your availability for a 30-minute sync this week.",
        "Good luck with your exam today! You've got this.",
        "Just wanted to say thank you for all your support on the release.",
        "Let me know if you need any assistance with the documentation review.",
        "Hope your week is off to a great start. Talk to you soon."
    ]

    for i, tmpl in enumerate(conversational_templates):
        for g in greetings:
            for n in names[:5]:
                for c in closings[:3]:
                    text = f"{g} {n},\n\n" + tmpl.format(name=n) + f"\n\n{c},\n{random.choice(names)}"
                    records.append({
                        'combined_text': text,
                        'clean_label': 0,
                        'category': 'benign_conversational',
                        'template_group_id': f"conv_{i}"
                    })

    # Short single-line conversational messages
    short_conversational = [
        "hi how are you",
        "hello how are you doing",
        "hi how are you?",
        "hello, how are you?",
        "thanks for your message",
        "see you tomorrow",
        "congratulations!",
        "meeting reminder for 3pm",
        "sounds good thanks",
        "perfect, see you then",
        "thanks a lot!",
        "got it, thank you",
        "let's meet at noon",
        "happy birthday!",
        "thank you so much",
        "yes that works for me",
        "great work on the release",
        "talk to you later"
    ]
    for i, msg in enumerate(short_conversational):
        records.append({
            'combined_text': msg,
            'clean_label': 0,
            'category': 'benign_short_conversational',
            'template_group_id': f"short_conv_{i}"
        })

    # =========================================================================
    # 2. CATEGORY B: LEGITIMATE NEWSLETTERS & CAREER NOTIFICATIONS (Label 0)
    # =========================================================================
    newsletter_topics = ["Python Weekly", "Cloud Architecture Digest", "Cybersecurity Trends", "Data Science Journal", "AI Weekly Review", "Campus Career Digest", "ACM TechNews", "Developer Roundup", "Open Source Monthly", "Engineering Bulletin"]
    newsletter_templates = [
        "Welcome to this week's edition of {topic}.\n\nIn this issue:\n- Modern microservice patterns and distributed tracing\n- Optimizing database query performance\n- Upcoming open-source webinars and community events\n\nRead the full issue online at your convenience. You are receiving this email because you subscribed to {topic}. To manage your preferences or unsubscribe, use your account dashboard.",
        "Welcome to {topic} issue #{num}.\nHighlights:\n- Optimizing database query performance\n- Modern async microservices architecture\n- Engineering best practices for reliability\nRead more at our website. To manage your preferences, visit your settings.",
        "Your weekly roundup from {topic} is here.\n\nHighlights:\n- Key takeaways from the annual developer conference\n- New research on neural network efficiency\n- Career advice: transitioning to senior engineering roles\n\nThanks for being a valued reader of our publication. To unsubscribe, click your profile settings.",
        "New Job Recommendations for You from Career Portal.\n\nBased on your profile, here are new positions matching your skills:\n1. Software Engineer - Cloud Services (Full-time)\n2. DevOps Specialist - Infrastructure Team\n3. Data Analyst - Business Intelligence\n\nView these opportunities on the portal. No action is required if you are not currently seeking new roles.",
        "University Announcement: Campus Seminar Series.\n\nPlease join us this Friday at 3:00 PM in the auditorium for a guest lecture on 'Future Trends in Computing'. All faculty, staff, and students are welcome to attend. Coffee and light refreshments will be served.",
        "Developer Community Digest: Releases & Updates.\n\nThis month's community spotlight features open-source contributions, upcoming hackathons, and tutorials on building responsive web applications. Check out the project repositories and contribute to ongoing discussions.",
        "{topic} Newsletter: Week in Review.\nHere are the top stories curated by our editorial team:\n- Advances in distributed systems and consensus protocols\n- Web performance optimization case study\n- Community open-source highlights\nEnjoy reading!",
        "Monthly Tech Bulletin - {topic}.\nCatch up on the latest tutorials, podcasts, and community discussions. No action is required. Thank you for subscribing."
    ]

    for i, tmpl in enumerate(newsletter_templates):
        for topic in newsletter_topics:
            for num in range(101, 150):
                text = tmpl.format(topic=topic, num=num)
                records.append({
                    'combined_text': text,
                    'clean_label': 0,
                    'category': 'benign_newsletters',
                    'template_group_id': f"news_{i}_{topic}"
                })

    # =========================================================================
    # 3. CATEGORY C: LEGITIMATE TRANSACTIONAL & NOTIFICATIONS (Label 0)
    # =========================================================================
    transactional_templates = [
        "Your order #{order_id} has shipped. No action is required.\n\nCarrier: Express Courier\nTracking Number: TRK-{order_id}-EXP\nEstimated Delivery: 2 business days.\nThank you for your business.",
        "Your order has shipped. No action is required.\nTracking is available in your account dashboard.",
        "Your order has shipped! We're packing your items carefully and they will arrive soon. No action is required on your part.",
        "Order Confirmation: Thank you for your order #{order_id}.\nWe have received your order and are currently processing it. You will receive another update with tracking info when it ships.\nTotal: $34.50. Paid via Visa ending in 4128. No action is required.",
        "Your package has been delivered.\nYour delivery for order #{order_id} was left at the front porch. We hope you enjoy your purchase! No action is required.",
        "Shipping Notification: Your package is on its way!\n\nTracking Number: TRK-{order_id}-US\nCarrier: Standard Express Logistics\nEstimated Delivery: 2-3 business days\n\nYou can track the delivery progress through our mobile app or website. Thank you for shopping with us.",
        "Appointment Confirmation: Scheduled Visit.\n\nYour appointment is confirmed for Tuesday at 10:30 AM with Dr. {name}.\nLocation: Suite 400, Medical Plaza.\nPlease arrive 10 minutes early to complete any check-in paperwork. To reschedule, contact our office directly.",
        "Application Confirmation: We received your submission.\n\nThank you for applying for the position of Systems Engineer. Our recruitment team is currently reviewing your resume and credentials. If your qualifications match our current openings, we will contact you for an initial interview.",
        "Subscription Invoice / Receipt.\n\nYour monthly subscription for Professional Plan has been renewed.\nInvoice Number: INV-2026-{order_id}\nBilling Period: Current Month\nAmount Paid: $19.00\nPayment Status: Paid\n\nYou can view all past invoices in your billing settings. No action is required.",
        "Meeting Invitation: Project Review and Sprint Planning.\n\nDate: Thursday, 2:00 PM - 3:00 PM\nLocation: Conference Room B & Online Video Link\nOrganizer: {name}\nAgenda:\n1. Sprint deliverables review\n2. Q3 roadmap prioritization\n3. Q&A\n\nPlease accept or decline the calendar invite.",
        "Receipt for your payment.\nThank you for your payment of $15.00 for order #{order_id}. Your transaction is complete and your receipt is attached. No action is required."
    ]

    for i, tmpl in enumerate(transactional_templates):
        for oid in range(1001, 1150):
            text = tmpl.format(order_id=oid, name=random.choice(names))
            records.append({
                'combined_text': text,
                'clean_label': 0,
                'category': 'benign_transactional',
                'template_group_id': f"trans_{i}"
            })

    # =========================================================================
    # 4. CATEGORY D: LEGITIMATE SECURITY & ACCOUNT NOTIFICATIONS (Label 0)
    # (Contains security/verify/password/account keywords in BENIGN context)
    # =========================================================================
    legit_security_templates = [
        "Security Alert: Your password was successfully updated.\n\nThe password for your account was changed on {date}.\nIf you initiated this change, no further action is required.\nIf you did not make this change, please contact our support team immediately or visit our official help center.",
        "New Sign-In Detected on Your Account.\n\nWe detected a sign-in to your account from a new device:\nDevice: Chrome on Windows\nLocation: Seattle, WA, USA\nTime: {date}\n\nIf this was you, you can safely disregard this email. If this wasn't you, please secure your account by changing your password from your account settings.",
        "Two-Factor Authentication Enabled.\n\nTwo-factor authentication (2FA) has been successfully activated on your account. Your account security has been upgraded. Please keep your backup recovery codes in a safe place. Our team will never ask you for your 2FA codes.",
        "We take account security seriously.\nNever share your password, OTP, or verification code with anyone. Our customer support team will never ask for your password via email or phone.",
        "Your email address verification was completed successfully.\n\nThank you for verifying your primary contact email address. Your account profile is now fully verified and active. You can now access all standard member features.",
        "Routine Security Checkup Summary.\n\nWe recently reviewed your account security settings. All active sessions and linked devices are within normal parameters. No unauthorized activity was detected. Thank you for keeping your account secure."
    ]

    dates = ["August 15, 2026 at 10:24 AM", "August 28, 2026 at 3:15 PM", "August 29, 2026 at 8:40 AM", "August 30, 2026 at 11:12 AM"]
    for i, tmpl in enumerate(legit_security_templates):
        for d in dates:
            for rep in range(15):
                text = tmpl.format(date=d)
                records.append({
                    'combined_text': text,
                    'clean_label': 0,
                    'category': 'legit_security_notification',
                    'template_group_id': f"legit_sec_{i}"
                })

    # =========================================================================
    # 5. CATEGORIES E-P: PHISHING, CREDENTIAL HARVESTING & THREAT ATTACKS (Label 1)
    # =========================================================================
    phishing_templates = [
        # E. Credential Harvesting with and without URLs
        "URGENT: Your account access has been suspended due to suspicious activity.\nTo restore your account immediately, visit the verification portal and submit your username and password:\nhttp://account-security-update.example/login\nFailure to verify within 24 hours will result in permanent account termination.",
        "Security Alert: Your account is scheduled for deletion.\nWe detected multiple unauthorized sign-in attempts. You must provide your password and OTP immediately to verify your identity:\nUsername: [Required]\nPassword: [Required]\nPIN/OTP: [Required]\nSubmit your details now to prevent permanent deletion.",
        "Your account will be permanently suspended unless you provide your password and OTP immediately.",
        "Visit the application portal and submit your password and OTP immediately.",
        "Your account will be suspended today unless you verify your password and OTP.",
        "Immediate Action Required: Confirm your login credentials.\nOur security system flagged unusual transactions. Reply to this email with your account password and security questions to unfreeze your balance.",
        "Payment Failed - Immediate Verification Required.\nYour recent payment could not be processed. Update your billing credentials and credit card details now:\nhttp://billing-update-portal.example/payment\nYour subscription will be cancelled immediately if not resolved.",
        "Invoice Overdue: Immediate Legal Action Warning.\nPlease review the attached invoice #88421 and submit your banking credentials and card PIN at http://invoice-payment-service.example/pay to avoid collection proceedings.",
        "Delivery Exception Notice: Package On Hold.\nYour shipment cannot be delivered due to an incorrect address. Confirm your identity and pay the $1.99 redelivery fee with your card details at http://express-package-redelivery.example/verify.",
        "Job Offer Acceptance: Submit Banking and Tax Credentials.\nCongratulations! You have been selected for the Remote Data Specialist position. To finalize your employment contract, enter your social security number, bank account details, and online banking password at http://careers-portal-onboarding.example/verify.",
        "Final Warning: Your mailbox is full and will be disabled.\nYour email storage has exceeded its quota limit (99.8%). Click here to upgrade storage and confirm your mailbox password: http://mail-server-quota-update.example/auth",
        "IRS / Tax Refund Notification: Claim Your $1,250.00 Refund.\nOur records indicate you are eligible for an immediate tax refund. Enter your full name, SSN, bank routing number, and login credentials at http://tax-refund-claim-portal.example/refund to receive direct deposit.",
        "Important Security Update: Re-authenticate your corporate account.\nDue to system migration, all employees must re-enter their network credentials and two-factor OTP tokens at http://corp-sso-migration-portal.example/login before the end of the day.",
        "Prize Claim Notification: You have won $50,000 in our international lottery.\nTo claim your winnings, provide your bank account credentials, password, and national identity card number to our agent immediately."
    ]

    for i, tmpl in enumerate(phishing_templates):
        for rep in range(40):
            records.append({
                'combined_text': tmpl,
                'clean_label': 1,
                'category': 'phishing_adversarial_and_direct',
                'template_group_id': f"phish_aug_{i}"
            })

    df_aug = pd.DataFrame(records)
    out_path = os.path.join(output_dir, 'augmented_robustness_corpus.csv')
    df_aug.to_csv(out_path, index=False)
    print(f"✅ Generated {len(df_aug):,} curated augmented email samples -> {out_path}")
    print(df_aug['clean_label'].value_counts())

if __name__ == '__main__':
    generate_augmented_email_corpus()
