
pub const LOG_KERNEL: bool = true;
pub const LOG_REST: bool = true;

#[macro_export]
macro_rules! log {
    ($category: ident, $($x:tt)*) => {
        if $crate::log::$category {
            print!($($x)*)
        }
    };
    ($($x:tt)*) => {
        if $crate::log::LOG_REST {
            print!($($x)*)
        }
    }
}