use crate::core::device::RRDevice;

pub trait GpuResource {
    unsafe fn destroy_gpu(&mut self, rrdevice: &RRDevice);
    fn resource_name(&self) -> &'static str;
}

pub unsafe fn destroy_all_in_reverse(resources: &mut [&mut dyn GpuResource], rrdevice: &RRDevice) {
    destroy_all_in_reverse_with(resources, |resource| resource.destroy_gpu(rrdevice));
}

fn destroy_all_in_reverse_with(
    resources: &mut [&mut dyn GpuResource],
    mut destroy: impl FnMut(&mut dyn GpuResource),
) {
    for resource in resources.iter_mut().rev() {
        log!("Destroying {}", resource.resource_name());
        destroy(&mut **resource);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;

    struct MockResource {
        name: &'static str,
        destroyed_order: Rc<RefCell<Vec<&'static str>>>,
    }

    impl GpuResource for MockResource {
        unsafe fn destroy_gpu(&mut self, _rrdevice: &RRDevice) {
            self.destroyed_order.borrow_mut().push(self.name);
        }

        fn resource_name(&self) -> &'static str {
            self.name
        }
    }

    #[test]
    fn destroys_resources_in_reverse_order() {
        let destroyed_order: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));

        let mut first = MockResource {
            name: "first",
            destroyed_order: destroyed_order.clone(),
        };
        let mut second = MockResource {
            name: "second",
            destroyed_order: destroyed_order.clone(),
        };
        let mut resources: [&mut dyn GpuResource; 2] = [&mut first, &mut second];

        destroy_all_in_reverse_with(&mut resources, |resource| {
            destroyed_order.borrow_mut().push(resource.resource_name())
        });

        assert_eq!(destroyed_order.borrow().as_slice(), ["second", "first"]);
    }
}
