import './Header.css';

function Header() {
  return (
    <header className="top-nav">
      <div className="nav-content">
        <div className="brand">
          <span className="brand-icon">📄</span>
          <span className="brand-name">DocuAssist</span>
        </div>
        <nav className="nav-links">
          <a href="#features">Features</a>
          <a href="#pricing">Pricing</a>
          <a href="#support">Support</a>
        </nav>
        <a className="cta-button" href="#get-started">
          Get Started
        </a>
      </div>
    </header>
  );
}

export default Header;
