use std::fmt;

#[derive(Debug)]
pub enum Error {
    InvalidParameter(String),
    LoadError(String),
    GenerateError(String),
    BackendError(String),
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidParameter(s) => write!(f, "Invalid parameter: {}", s),
            Error::LoadError(s) => write!(f, "Load error: {}", s),
            Error::GenerateError(s) => write!(f, "Generate error: {}", s),
            Error::BackendError(s) => write!(f, "Backend error: {}", s),
            Error::Unsupported(s) => write!(f, "Unsupported: {}", s),
        }
    }
}

impl std::error::Error for Error {}
