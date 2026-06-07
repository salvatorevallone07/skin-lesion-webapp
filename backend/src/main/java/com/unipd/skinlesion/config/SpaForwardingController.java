package com.unipd.skinlesion.config;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

/** Forwards non-API, non-static routes to the Angular index.html (SPA support). */
@Controller
public class SpaForwardingController {

    @GetMapping(value = {"/", "/{path:[^\\.]*}"})
    public String forward() {
        return "forward:/index.html";
    }
}
